"""Replay public lossless evidence without the private recording harness.
Usage: python experiments/2026-11-16-guarded-fp32-baseline/analysis/replay_public.py
"""
import csv,hashlib,io,json,math,struct,tempfile,zipfile
from pathlib import Path,PurePosixPath
from independent_statistics import compute
ROOT=Path(__file__).resolve().parents[3];E='2026-11-16-guarded-fp32-baseline';RID='20261117-cuda5-fp32-repair02'
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

def gsp_documents(data):
 assert data[:8]==b'GSPIDX01'
 n,v,z=struct.unpack_from('<IIQ',data,8);ids=struct.unpack_from('<%dQ'%n,data,24);offset=24+8*n
 offsets=struct.unpack_from('<%dQ'%(n+1),data,offset);offset+=8*(n+1)
 terms=struct.unpack_from('<%dI'%z,data,offset);impacts=data[offset+4*z:offset+5*z]
 offset+=5*z;dictionary=[]
 for _ in range(v):
  size=struct.unpack_from('<I',data,offset)[0];offset+=4
  assert offset+size<=len(data);dictionary.append(data[offset:offset+size].decode('utf-8'));offset+=size
 assert offset==len(data) and len(set(dictionary))==v
 assert len(impacts)==z and len(set(ids))==n and offsets[0]==0 and offsets[-1]==z
 documents=[]
 for d in range(n):
  row=list(zip(terms[offsets[d]:offsets[d+1]],impacts[offsets[d]:offsets[d+1]]))
  assert all(0<=t<v for t,w in row) and all(a[0]<b[0] for a,b in zip(row,row[1:]))
  documents.append(row)
 return ids,documents


def main():
 run=ROOT/'results'/E/RID;manifest=load(run/'manifest.json');assert manifest['status']=='complete'
 protocol=load(ROOT/'experiments'/E/'configs/protocol.json');full_environment=None;boundary=None
 for name,item in manifest['artifacts'].items():assert sha((run/name).read_bytes())==item['sha256']
 attempts=load(run/'attempts.json');assert len(attempts)>=42
 reference_run=ROOT/'results/2026-11-15-full-collection-baselines/20261115-cuda5-reference01';reference_manifest=load(reference_run/'manifest.json')
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
 assert all(len(v)==10 for v in oracle.values());groups=scores=0;paths={k:[] for k in ['range','mono','f32']};accepted_slots=set()
 with zipfile.ZipFile(run/'raw-evidence.zip') as z,tempfile.TemporaryDirectory() as td:
  assert z.testzip() is None
  for name,item in manifest['raw_members'].items():assert sha(z.read(name))==item['sha256']
  candidates=[json.loads(x) for x in z.read('analysis/candidate-rows.jsonl').decode().splitlines()];docs={r['original_id']:r['terms'] for r in candidates};positions={r['position']:r['original_id'] for r in candidates}
  for q,values in oracle.items():
   for d,score in values:assert sum(queries[q].get(t,0)*w for t,w in docs[d])==score
  # Keep registered process order from the completed execution manifest.
  execution=load(run/'execution-manifest.json');completed=execution['completed'];reported=execution['analysis'];ordered={x['tag']:i for i,x in enumerate(completed)};by_tag={x['tag']:x for x in completed}
  for attempt in sorted(attempts,key=lambda x:ordered[x['tag']]):
   tag=attempt['tag'];runtime=attempt['resolved_runtime'];cfg=runtime['config'];actual=runtime['behavior'];label={'cuda-range-v2':'range','gpusparse-u64-canonical':'mono','gpusparse-fp32guarded-canonical':'f32'}[cfg['mode']]
   if tag.startswith('reject-'):
    assert attempt['status']=='SPEC_VIOLATION' and len(attempt['runtime_delta'])==1;continue
   assert attempt['status']=='EXECUTED' and cfg==attempt['expected'] and actual==attempt['expected_behavior']
   full=tag.startswith(('correctness-','timing-'))
   if full:
    assert cfg['docs']==8841823 and cfg['query_count']==500 and cfg.get('k',cfg.get('requested_k'))==10 and cfg['batch_size']==1 and cfg['cpu_affinity']=='0'
    assert cfg['warmup_rounds']==(3 if tag.startswith('timing-') else 0) and cfg['repeats']==(20 if tag.startswith('timing-') else 1)
    roles={x['role']:x['sha256'] for x in attempt['inputs']}
    for role,key in [('canonical_gsp','gsp'),('queries','queries'),('reference','oracle'),('query_split','query_split')]:assert roles[role]==protocol['inputs'][key]['sha256']
    assert attempt['split']=='calibration' and actual['tie_order']=='score_desc_original_numeric_id_asc' and actual['eligibility']=='all_documents'
    if boundary is None:boundary=attempt['timing_boundary']
    else:assert boundary==attempt['timing_boundary']
    env=attempt['environment'];gpu=[value.strip() for value in next(csv.reader(env['gpu']['stdout'].splitlines()))];assert gpu[1]==protocol['target_uuid']
    stable={key:env[key] for key in ['host','platform','python','env','compiler','cuda_compiler']};stable['gpu_identity_driver_memory']=gpu[:4]
    if full_environment is None:full_environment=stable
    else:close(stable,full_environment)
    if label=='f32':assert cfg['accumulator_dtype']==cfg['query_gpu_dtype']==cfg['document_gpu_dtype']=='torch.float32' and cfg['floating_score_limit_exclusive']==2**24 and cfg['native_scoring_reused'] is True
    expected=oracle
   else:
    roles={x['role']:x for x in attempt['inputs']};remote_run=PurePosixPath(attempt['cwd'])/'results'/E/RID
    raw_gsp=z.read(str(PurePosixPath(roles['canonical_gsp']['path']).relative_to(remote_run)));raw_query=z.read(str(PurePosixPath(roles['queries']['path']).relative_to(remote_run)))
    assert sha(raw_gsp)==roles['canonical_gsp']['sha256'] and sha(raw_query)==roles['queries']['sha256']
    ids,documents=gsp_documents(raw_gsp);expected={}
    for line in raw_query.decode().splitlines():
     fields=line.split();weights={int(t):int(w) for t,w in (x.split(':') for x in fields[1:])}
     expected[fields[0]]=sorted([(d,sum(weights.get(t,0)*w for t,w in doc)) for d,doc in zip(ids,documents)],key=lambda pair:(-pair[1],pair[0]))[:min(10,len(ids))]
   wanted=[(phase,n,q) for phase,count in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for n in range(count) for q in expected]
   actual_groups={};observed=[]
   for row in rows(z.read('raw/'+tag+'/top.tsv'),'\t'):
    key=(row['phase'],int(row['round']),row['query_id'])
    if key not in actual_groups:actual_groups[key]=[];observed.append(key)
    assert int(row['rank'])==len(actual_groups[key])+1;actual_groups[key].append((int(row['original_id']),int(row['score'])));scores+=1
    if full and label in ['mono','f32']:assert positions[int(row['internal_doc_position'])]==int(row['original_id'])
   assert observed==(wanted if cfg['docs'] else []);assert all(actual_groups.get(key,[])==expected[key[2]] for key in wanted);groups+=len(wanted)
   data=z.read('raw/'+tag+'/timing.csv');assert [(r['phase'],int(r['round']),r['query_id']) for r in rows(data)]==wanted
   if tag.startswith('timing-'):
    entry=by_tag[tag];monitor=entry['monitor'];assert not monitor['sampling_errors'] and monitor['owned_context_seen']
    frames=[json.loads(line) for line in z.read('logs/'+tag+'.monitor.jsonl').decode().splitlines()]
    assert frames and all('error' not in frame and frame['gpu'][0]==protocol['target_uuid'] for frame in frames) and len(frames)==monitor['samples']
    assert max(sum(c['used_mib'] for c in f['contexts'] if c['owned']) for f in frames)==monitor['sampled_own_peak_mib']
    assert max(int(f['gpu'][1]) for f in frames)==monitor['sampled_board_peak_mib']
    foreign={context['pid'] for frame in frames for context in frame['contexts'] if not context['owned']};assert foreign==set(monitor['external_pids'])
    assert entry['eligible']==(not foreign)
    if not entry['eligible']:continue # Answers above were checked; contaminated timings never enter statistics.
    parts=tag.split('-resource');assert len(parts) in [1,2] and (len(parts)==1 or parts[1] in ['1','2','3']);slot=parts[0];assert slot not in accepted_slots and slot in {'timing-r%d-%s'%(r,label) for r in range(3) for label in ['range','mono','f32']};accepted_slots.add(slot)
    memory={'sampled_own_peak_mib':monitor['sampled_own_peak_mib'],'sampled_board_peak_mib':monitor['sampled_board_peak_mib']}
    if label=='range':memory['tracked_application_bytes']=int(next(rows(data))['device_allocation_bytes'])
    else:
     native=json.loads(z.read('raw/'+tag+'/native-validation.json'));memory['torch_allocator_peak_after_setup_bytes']=native['torch_allocator_peak_bytes'];memory['torch_reserved_peak_after_setup_bytes']=native['torch_allocator_reserved_peak_bytes']
    close(memory,reported['summary'][label]['memory'][int(slot.split('-')[1][1:])])
    path=Path(td)/(tag+'.csv');path.write_bytes(data);paths[label].append(path)
  physical=json.loads(z.read('analysis/physical-arithmetic.json'));assert physical['status']=='PASS' and len({e['tag'] for e in physical['entries']})==8
  for item in physical['entries']:
   raw=z.read(item['public_ptx']);assert sha(raw)==item['sha256'];dtype='f32' if item['label']=='f32' else 'u64'
   lines=[line for line in raw.decode().splitlines() if ('atom.' in line or 'red.' in line) and '.add.' in line];assert lines and all('.add.'+dtype+' ' in line for line in lines)
  assert json.loads(z.read('analysis/postrun-library-check.json'))['equal_to_pre_run']
  assert json.loads(z.read('logs/packages.stdout.log'))==json.loads(z.read('logs/packages-after.stdout.log'))
  assert len(accepted_slots)==sum(len(v) for v in paths.values())==9
  result=compute(paths,list(queries));reported=load(run/'execution-manifest.json')['analysis']
  close(result['comparisons'],reported['comparisons'])
  for label,values in result['summary'].items():
   for key,value in values.items():close(value,reported['summary'][label][key])
 print(json.dumps(dict(success=True,groups=groups,scores=scores,oracle_scores_checked=5000,independent_bootstrap_draws=10000,qualification='500calibration queries; Guarded canonical FP32 and canonical U64 GPUSparse adaptations, not unmodified native. Partial source rows are remotely hash-bound, not a local full-index membership proof.')))
if __name__=='__main__':main()
