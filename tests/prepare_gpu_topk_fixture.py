"""Independent GSP writer and Python oracle; no production scorer is imported."""
import argparse,json,random,struct
from pathlib import Path

def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rng=random.Random(20261111);metadata=[]
    queries={'empty':{},'term0':{0:32},'term1':{1:32},'wide':{t:32 for t in range(33)},'mixed':{0:1,3:17,32:31}}
    with (root/'queries.tsv').open('x',encoding='utf-8',newline='\n') as f:
        for name,weights in queries.items():f.write(name+''.join('\t%d:%d'%item for item in weights.items())+'\n')
    for n in [0,1,17,255,256,257,1025]:
        ids=list(range(1000,1000+n));rng.shuffle(ids)
        docs=[{} if d%11==0 else {t:rng.choice([1,15,16,31,240,255]) for t in range(33) if rng.random()<.25} for d in range(n)]
        if n>1:docs[0]=docs[-1]={t:255 for t in range(33)}
        offsets=[0];terms=[];impacts=[]
        for doc in docs:
            for t,v in sorted(doc.items()):terms.append(t);impacts.append(v)
            offsets.append(len(terms))
        with (root/('fixture-%d.gsp'%n)).open('xb') as f:
            f.write(b'GSPIDX01'+struct.pack('<IIQ',n,33,len(terms)))
            for values,fmt in [(ids,'Q'),(offsets,'Q'),(terms,'I'),(impacts,'B')]:
                for value in values:f.write(struct.pack('<'+fmt,value))
            for t in range(33):
                word=('t%d'%t).encode();f.write(struct.pack('<I',len(word))+word)
        expected={q:sorted([(original,sum(v*weights.get(t,0) for t,v in doc.items())) for original,doc in zip(ids,docs)],key=lambda x:(-x[1],x[0])) for q,weights in queries.items()}
        (root/('expected-%d.json'%n)).write_text(json.dumps(expected,indent=2)+'\n',encoding='utf-8')
        metadata.append({'docs':n,'queries':len(queries),'depths':[0,1,10,100,128]})
    (root/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(prepare(a.directory)))
