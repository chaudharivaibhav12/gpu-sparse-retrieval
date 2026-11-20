import csv,gzip,lzma,hashlib,json,math,statistics,subprocess,zipfile,io
from pathlib import Path
R=Path(__file__).resolve().parents[3]
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def stream(p):
 return (gzip.open if p.suffix=='.gz' else lzma.open if p.suffix=='.xz' else open)(p,'rt',newline='')
with zipfile.ZipFile(R/'results/2026-11-15-full-collection-baselines/20261115-cuda5-reference01/raw-evidence.zip') as z:
 names=[n for n in z.namelist() if n.endswith('full-top10.tsv')];assert len(names)==1,names
 gold={}
 for row in csv.DictReader(io.StringIO(z.read(names[0]).decode()),delimiter='\t'):
  gold.setdefault(row['query_id'],[]).append((int(row['original_id']),int(row['score'])))
assert len(gold)==500
studies=[('2026-11-18-prune-seed-select','20261118-cuda5-ts-01'),('2026-11-18-prune-term-bounds','20261118-cuda5-tb-01'),('2026-11-18-range-pruning','20261117-cuda5-prune-04'),('2026-11-18-bmp-full-collection','20261118-cuda5-bmp-02'),('2026-11-18-full-profile','20261118-cuda5-profile-01'),('2026-11-15-range-batching','20261117-cuda5-full-03')]
reports=[]
for e,rid in studies:
 run=R/'results'/e/rid;m=read(run/'manifest.json'); report={'experiment':e,'run':rid,'recording':m.get('recording'),'measured_source':m.get('source_commit'),'manifest_keys':list(m),'file_checks':0,'source_bindings':0,'ranking_groups':0,'timing_medians':{},'paired_points':{}}
 for key,table in m.items():
  if not isinstance(table,dict):continue
  for name,proof in table.items():
   if not isinstance(proof,dict) or 'sha256' not in proof:continue
   p=run/name
   if p.is_file():
    b=p.read_bytes();assert sha(b)==proof['sha256'],p
    if 'bytes' in proof:assert len(b)==proof['bytes'],p
    report['file_checks']+=1
 for name,digest in m.get('build',{}).get('source_sha256',{}).items():
  if not m.get('source_commit'):continue
  b=subprocess.run(['git','show',m['source_commit']+':'+name],cwd=R,check=True,capture_output=True).stdout
  assert sha(b)==digest,(rid,name);report['source_bindings']+=1
 timing={}
 for p in sorted((run/'raw').rglob('*')):
  if not p.is_file():continue
  if any(x in p.parts for x in ('ineligible','interrupted')):continue
  if p.suffixes[-2:]==['.answers','.json']:
   receipt=read(p);h=hashlib.sha256(b'phase\tround\tquery_id\trank\toriginal_id\tscore\n');total=0
   for group in receipt['groups']:
    body=gzip.decompress((run/'raw/answers'/(group['body_sha256']+'.tsv.gz')).read_bytes())
    assert sha(body)==group['body_sha256'];lines=body.decode().splitlines(keepends=True);assert len(lines)==group['rows']
    groups={}
    for line in lines:
     q,rank,doc,score=line.split();v=groups.setdefault(q,[]);assert int(rank)==len(v)+1;v.append((int(doc),int(score)))
     full=(str(group['phase'])+'\t'+str(group['round'])+'\t'+line).encode();h.update(full);total+=len(full)
    assert set(groups)==set(gold)
    for q,values in groups.items():assert values==gold[q],(p,q)
    report['ranking_groups']+=len(groups)
   assert h.hexdigest()==receipt['top_sha256'] and total+len(b'phase\tround\tquery_id\trank\toriginal_id\tscore\n')==receipt['top_bytes'],p
  if p.name.endswith(('.tsv.xz','.tsv.gz')) and not 'ineligible' in p.parts and not 'interrupted' in p.parts:
   with stream(p) as f:
    reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames
    if not fields or not {'query_id','rank','score'}.issubset(fields):continue
    idkey='original_id' if 'original_id' in fields else 'document_id' if 'document_id' in fields else None
    if not idkey:continue
    groups={}
    for row in reader:
     key=(row.get('phase'),row.get('round'),row.get('layout'),row['query_id'])
     values=groups.setdefault(key,[]);assert int(row['rank'])==len(values)+1,(p,key)
     values.append((int(row[idkey]),int(row['score'])))
    for key,values in groups.items():assert values==gold[key[-1]],(p,key,values,gold[key[-1]])
    report['ranking_groups']+=len(groups)
  if p.name.startswith('timing-r') and p.name.endswith(('.csv','.csv.gz','.csv.xz')):
   with stream(p) as f:
    rows=list(csv.DictReader(f))
   if not rows or 'query_wall_ms' not in rows[0]:continue
   assert len(rows)==11500,(p,len(rows));data={}
   for row in rows:
    if row['phase']=='measured':data.setdefault(row['query_id'],[]).append(float(row['query_wall_ms']))
   assert set(data)==set(gold) and all(len(v)==20 for v in data.values())
   stem=p.name.split('.csv')[0]
   receipt=read(p.parent/(stem+'.receipt.json'));assert receipt['verdict']=='accepted' and receipt['exit_code']==0
   assert receipt['gpu_eligible'] and receipt['cpu_sampling']['errors']==0 and receipt['cpu_sampling']['other_cores_mean']<=4,(p,receipt['cpu_sampling'])
   runtime=read(p.parent/(stem+'.receipt.runtime')/'runtime.json')
   for kind,expected in receipt['expected'].items():
    assert all(runtime[kind].get(k)==v for k,v in expected.items()),p
   report['runtime_gate_qualification']='Author gate validates expected subset; extra resolved fields are retained but not rejected. Future team-runtime runs must freeze complete state.'
   name=stem.split('-',2)[2];timing.setdefault(name,[]).append({q:statistics.median(v) for q,v in data.items()})
  if p.name=='samples.tsv' and p.parent.name.startswith('timing-r') and p.parent.name.endswith('bmp-b32.bmp'):
   data={}
   for row in csv.DictReader(p.open(newline=''),delimiter='\t'):
    if row['phase']=='measured':data.setdefault(row['query_id'],[]).append(int(row['elapsed_ns'])/1e6)
   assert set(data)==set(gold) and all(len(v)==20 for v in data.values())
   timing.setdefault('bmp-b32',[]).append({q:statistics.median(v) for q,v in data.items()})
 for name,ps in timing.items():report['timing_medians'][name]=statistics.median(v for d in ps for v in d.values())
 if 'v2-w1024' in timing:
  for name,ps in timing.items():
   if name=='v2-w1024':continue
   assert len(ps)==len(timing['v2-w1024'])==3
   report['paired_points']['v2-w1024 / '+name]=math.exp(statistics.fmean(math.log(timing['v2-w1024'][i][q]/ps[i][q]) for i in range(3) for q in gold))
 saved=run/'analysis/statistics.json'
 if saved.exists():
  s=read(saved)
  for key,point in report['paired_points'].items():
   if key in s.get('contrasts',{}):assert math.isclose(point,s['contrasts'][key]['ratio'],rel_tol=1e-12)
 reports.append(report);(OUT/(e+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
(OUT/'summary.json').write_text(json.dumps(reports,indent=2))
