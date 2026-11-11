"""Extend the independent parent oracle with interval-boundary cases."""
import argparse,importlib.util,json,random,struct
from pathlib import Path

def prepare(root):
    spec=importlib.util.spec_from_file_location('parent_fixture',Path(__file__).with_name('prepare_fullbound_fixture.py'))
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    cases=parent.prepare(root)
    queries={}
    for line in (root/'queries.tsv').read_text(encoding='utf-8').splitlines():
        fields=line.split();queries[fields[0]]={int(a):int(b) for a,b in (token.split(':') for token in fields[1:])}
    for seed in [15,16]:
        n=1025;label='interval-seed%d-1025'%seed;rng=random.Random(20261111+seed)
        ids=list(range(1000,1000+n));rng.shuffle(ids);ids[0]=2**64-1;ids[-1]=0
        vals=[seed if d<512 or d==n-1 else [0,1,15,16,17,31,32,255][((d-512)//32)%8] for d in range(n)]
        docs=[{0:v} if v else {} for v in vals];offsets=[0];terms=[];impacts=[]
        for doc in docs:
            for t,v in doc.items():terms.append(t);impacts.append(v)
            offsets.append(len(terms))
        with (root/('fixture-'+label+'.gsp')).open('xb') as f:
            f.write(b'GSPIDX01'+struct.pack('<IIQ',n,33,len(terms)))
            for values,fmt in [(ids,'Q'),(offsets,'Q'),(terms,'I'),(impacts,'B')]:
                for v in values:f.write(struct.pack('<'+fmt,v))
            for t in range(33):
                word=('t%d'%t).encode();f.write(struct.pack('<I',len(word))+word)
        expected={q:sorted([(original,sum(v*weights.get(t,0) for t,v in doc.items())) for original,doc in zip(ids,docs)],key=lambda x:(-x[1],x[0])) for q,weights in queries.items()}
        with (root/('expected-'+label+'.json')).open('x',encoding='utf-8') as f:json.dump(expected,f,indent=2);f.write('\n')
        cases.append(dict(label=label,docs=n,queries=5,depths=[0,10,128]))
    with (root/'precision-manifest.json').open('x',encoding='utf-8') as f:json.dump(cases,f,indent=2);f.write('\n')
    return cases

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(prepare(a.directory)))
