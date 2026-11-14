"""Independent Python-integer oracle above the FP32 exact-integer threshold."""
import json, struct
from pathlib import Path

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    ids=[2**64-1,2**63+5,0,7];vocab=4097
    docs=[{**{t:255 for t in range(4096)},4096:(2 if i==2 else 1)} for i in range(4)]
    queries={'above-fp32':{**{t:32 for t in range(4096)},4096:1},
             'empty':{},'tail':{4096:1},'first':{0:32},'tail32':{4096:32}}
    offsets=[0];terms=[];impacts=[]
    for row in docs:
        for term,value in row.items():terms.append(term);impacts.append(value)
        offsets.append(len(terms))
    with (root/'fixture-high.gsp').open('xb') as f:
        f.write(b'GSPIDX01'+struct.pack('<IIQ',len(ids),vocab,len(terms)))
        for values,fmt in [(ids,'Q'),(offsets,'Q'),(terms,'I'),(impacts,'B')]:
            for value in values:f.write(struct.pack('<'+fmt,value))
        for t in range(vocab):
            word=('t%d'%t).encode();f.write(struct.pack('<I',len(word))+word)
    with (root/'queries.tsv').open('x',encoding='utf-8',newline='\n') as f:
        for q,weights in queries.items():f.write(q+''.join(' %d:%d'%item for item in weights.items())+'\n')
    expected={q:sorted([(identity,sum(value*weights.get(term,0) for term,value in row.items()))
               for identity,row in zip(ids,docs)],key=lambda x:(-x[1],x[0])) for q,weights in queries.items()}
    assert expected['above-fp32']==[(0,33423362),(7,33423361),(2**63+5,33423361),(2**64-1,33423361)]
    with (root/'expected-high.json').open('x',encoding='utf-8') as f:json.dump(expected,f,indent=2);f.write('\n')
    return dict(label='high',docs=4,queries=5,depths=[1,10,128])
