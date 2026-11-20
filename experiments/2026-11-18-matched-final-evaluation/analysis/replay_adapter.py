"""Replay public CPU adapter validation with only Python's standard library."""
import collections,csv,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
E='2026-11-18-matched-final-evaluation'
def load_run(rid):
    folder=ROOT/'results'/E/rid;m=json.loads((folder/'manifest.json').read_text())
    archive=folder/'raw-evidence.zip';b=archive.read_bytes()
    assert hashlib.sha256(b).hexdigest()==m['archive']['sha256'] and len(b)==m['archive']['bytes']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(m['raw_members'])
        result={}
        for name,proof in m['raw_members'].items():
            b=z.read(name);assert hashlib.sha256(b).hexdigest()==proof['sha256'] and len(b)==proof['bytes'];result[name]=b
    return result
def replay():
    build=load_run('20261118-cuda5-adapter-build01')
    assert b'3 passed; 0 failed' in build['adapter-tests.stdout']
    source=(ROOT/'experiments'/E/'code/project_benchmark_matched.rs').read_bytes()
    # Git canonical LF bytes are the source measured on CIMS.
    source=source.replace(b'\r\n',b'\n')
    assert source==build['actual-adapter.rs']
    raw=load_run('20261118-cuda5-adapter-fixture01')
    docs=json.loads(raw['fixture-inputs/documents.json']);gold=json.loads(raw['fixture-inputs/expected.json'])
    queries={}
    for line in raw['fixture-inputs/prepared.pisa'].decode().splitlines():
        q,text=line.split(':',1);assert q not in queries;queries[q]=collections.Counter(text.split())
    assert len(queries)==12 and len(docs)==193 and set(queries)==set(gold)
    for q,weights in queries.items():
        assert all(0<w<=32 for w in weights.values()) and len(weights)<=128
        actual=sorted([[int(doc),sum(int(w)*weights.get('t'+t,0) for t,w in terms.items())] for doc,terms in docs.items()],key=lambda p:(-p[1],p[0]))
        assert actual==gold[q]
    canonical='fixture-validation-02/canonical-fixture'
    receipt=json.loads(raw[canonical+'/attempt.json'])
    assert receipt['status']=='EXECUTED' and receipt['runtime']=={'config':receipt['expected'],'behavior':receipt['expected_behavior']}
    groups={}
    for row in csv.DictReader(io.StringIO(raw[canonical+'/results.tsv'].decode()),delimiter='\t'):
        assert row['phase']=='measured' and row['round']=='0'
        v=groups.setdefault(row['query_id'],[]);assert int(row['rank'])==len(v)+1;v.append([int(row['document_id']),int(row['score'])])
    assert set(groups)==set(gold)
    for q,values in groups.items():assert values==gold[q][:10]
    for attempt in ['fixture-validation-01/deliberate-cpu-drift','fixture-validation-02/deliberate-cpu-drift']:
        receipt=json.loads(raw[attempt+'/attempt.json']);assert receipt['status']=='SPEC_VIOLATION' and set(receipt['runtime_delta'])=={'config.cpu'}
        assert receipt['runtime']['config']['cpu']==0 and receipt['expected']['cpu']==1
        for name in ['samples','results','monitoring']:
            data=raw[attempt+'/'+name+'.tsv'].decode();assert len(data.splitlines())==1 and not list(csv.DictReader(io.StringIO(data),delimiter='\t'))
    print(json.dumps({'status':'PASS','unit_tests':3,'independent_document_query_scores':193*12,'canonical_ranked_scores':120,'deliberate_rejections':2,'qualification':'Synthetic CPU correctness only; no performance or heldout claim'}))
if __name__=='__main__':replay()
