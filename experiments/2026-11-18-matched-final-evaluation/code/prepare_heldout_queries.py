"""Frozen cap32 preprocessing. Only call after the immutable method freeze exists."""
import collections,csv,hashlib,importlib.util,json,math,re,struct
from pathlib import Path
RAW_SHA='d4781e91329b81ea0b6fe2b34ad06ba5c0bb0c049e63c5ff7e4a996af88a0f8a'
SPLIT_SHA='650a121bf25d71beee4e2be8d05228ce8d700b2a866d8c8ba996e0ec53ce33d1'
CAL_SHA='150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def transform(text,vocabulary):
 counts=collections.Counter(re.findall(r'[^ \t\r\n\v\f]+',text));maximum=max(counts.values(),default=0)
 if maximum>=2**32:raise ValueError('Token count outside original uint32 domain')
 scale=f32(f32(32)/f32(maximum)) if maximum>32 else 1.0
 result=sorted((vocabulary[t],math.ceil(f32(f32(n)*scale)) if maximum>32 else n) for t,n in counts.items() if t in vocabulary)
 if len(result)>128 or any(not 1<=w<=32 for _,w in result) or 255*sum(w for _,w in result)>=2**24:raise ValueError('Unsupported common exact domain; stop without dropping query')
 return result

def prepare(repository,raw,split,calibration,out,freeze):
 repository=Path(repository);out=Path(out)
 # Check before opening the raw query source; no heldout content before freeze.
 frozen=json.loads(Path(freeze).read_text());assert frozen['kind']=='freeze' and frozen['payload']['evaluation_rule']['query_count']==6480
 assert digest(raw)==RAW_SHA and digest(split)==SPLIT_SHA and digest(calibration)==CAL_SHA
 meta=list(csv.DictReader(Path(split).open(),delimiter='\t'));ids=[x['query_id'] for x in meta]
 assert len(meta)==len(set(ids))==6980 and all(x['split']==('calibration' if i<500 else 'heldout') for i,x in enumerate(meta))
 assert [(int(x['hash']),x['query_id']) for x in meta]==sorted((int(x['hash']),x['query_id']) for x in meta)
 index=frozen['payload']['evaluation_rule']['gsp_path'];path=repository/'experiments/2026-11-18-bmp-full-collection/code/export_queries.py';spec=importlib.util.spec_from_file_location('vocabulary_only',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);docs,words=module.vocabulary(Path(index));assert docs==8841823 and len(words)==len(set(words))==28679;vocab={w:i for i,w in enumerate(words)}
 prepared={};unsupported=[];set_ids=set(ids)
 with Path(raw).open(encoding='utf-8') as f:
  for line in f:
   qid,sep,text=line.rstrip('\n').partition(':')
   if not sep or qid not in set_ids or qid in prepared:raise ValueError('Raw query membership or duplicate differs from frozen split')
   try:terms=transform(text,vocab)
   except ValueError as e:unsupported.append({'query_id':qid,'error':str(e)});terms=[]
   prepared[qid]=qid+''.join('\t%d:%d'%x for x in terms)+'\n'
 assert set(prepared)==set(ids)
 calibration_bytes=''.join(prepared[q] for q in ids[:500]).encode();assert hashlib.sha256(calibration_bytes).hexdigest()==CAL_SHA,'Preprocessing differs from original500calibration bytes'
 if unsupported:raise ValueError('Unsupported heldout/common-domain queries; no drop or tuning: '+json.dumps(unsupported))
 out.mkdir(parents=True,exist_ok=False);queries=out/'heldout-6480.tsv';pisa=out/'heldout-6480.pisa';queries.write_bytes(''.join(prepared[q] for q in ids[500:]).encode())
 with pisa.open('x',encoding='utf-8',newline='\n') as f:
  for q in ids[500:]:
   pairs=[tuple(map(int,t.split(':'))) for t in prepared[q].split()[1:]];f.write(q+': '+' '.join(words[t] for t,w in pairs for _ in range(w))+'\n')
 report={'queries':{'path':str(queries),'sha256':digest(queries)},'pisa':{'path':str(pisa),'sha256':digest(pisa)},'query_split_sha256':SPLIT_SHA,'raw_sha256':RAW_SHA,'calibration_reproduction_sha256':hashlib.sha256(calibration_bytes).hexdigest(),'heldout_count':6480,'calibration_count':500,'disjoint':set(ids[:500]).isdisjoint(ids[500:]),'all_supported':True,'policy':'Original all-token maxcount scaling with separate binary32 rounding before ceil; OOV counts affect scaling,known terms retained,numeric-term order'}
 (out/'preparation.json').write_text(json.dumps(report,indent=2));return report
