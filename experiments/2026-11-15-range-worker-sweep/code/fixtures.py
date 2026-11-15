"""Reuse the parent's independent fixture definitions without its direct-run harness."""
import importlib.util,json,random
from pathlib import Path

def prepare(root,repo):
 path=Path(repo)/'tests/check_range_bitpack.py';spec=importlib.util.spec_from_file_location('range_parent_fixtures',path);parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
 root=Path(root);root.mkdir(parents=True,exist_ok=False);rng=random.Random(20261114);manifest=[]
 cases=parent.fixtures()
 ids=[2**64-1,2**63+5,0,7]
 docs=[[(t,255) for t in range(127) for _ in range(17)]+[(127,2 if i==2 else 1)] for i in range(4)]
 cases['high-score']=(ids,docs,129)
 for label,(ids,docs,vocab) in cases.items():
  gsp=root/(label+'.gsp');parent.write_gsp(gsp,ids,docs,vocab);queries=parent.queries_for(label,vocab,rng)
  if label=='high-score':queries['above-fp32']={**{t:32 for t in range(127)},127:1}
  q=root/(label+'-queries.tsv')
  with q.open('x',encoding='utf-8',newline='\n') as f:
   for qid,terms in queries.items():f.write(qid+''.join(' %d:%d'%item for item in terms.items())+'\n')
  expected={qid:parent.expected_ranking(ids,docs,weights) for qid,weights in queries.items()}
  if label=='high-score':assert expected['above-fp32']==[(0,17617442),(7,17617441),(2**63+5,17617441),(2**64-1,17617441)]
  oracle=root/(label+'-expected.json')
  with oracle.open('x',encoding='utf-8') as f:json.dump(expected,f);f.write('\n')
  manifest.append(dict(label=label,gsp=str(gsp),queries=str(q),oracle=str(oracle),docs=len(ids),vocab=vocab,query_count=len(queries)))
 with (root/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2);f.write('\n')
 return manifest,parent
