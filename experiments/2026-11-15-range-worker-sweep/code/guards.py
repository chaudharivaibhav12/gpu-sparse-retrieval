"""Prepare negative inputs; caller must use a fail-closed gate with no continue file."""
from pathlib import Path

def output_paths(root,label,query_path):
 root=Path(root);csv=root/(label+'.csv');top=root/(label+'.top.tsv')
 assert csv.resolve()!=Path(query_path).resolve() and top.resolve()!=Path(query_path).resolve()
 assert not csv.exists() and not top.exists(),'Negative-test outputs must be fresh'
 return csv,top

def prepare(root,parent,fixture_root):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);fixture_root=Path(fixture_root)
 q=root/'too-many-terms.tsv';q.write_text('q'+''.join(' %d:32'%i for i in range(129))+'\n',encoding='utf-8')
 paths=[dict(label='too-many-terms',gsp=fixture_root/'big.gsp',queries=q,error='more terms than RBP_MAX_TERMS')]
 for label,text,error in [('duplicate-query-term','q 0:1 0:2\n','Bad cap32 query'),('out-of-range-term','q 99999:1\n','Bad cap32 query'),('weight-above-cap','q 0:33\n','Bad cap32 query')]:
  q=root/(label+'.tsv');q.write_text(text,encoding='utf-8');paths.append(dict(label=label,gsp=fixture_root/'ties.gsp',queries=q,error=error))
 gsp=root/'conservative-overflow.gsp';parent.write_gsp(gsp,[7],[[(0,1)]*65535],11);q=root/'overflow.tsv';q.write_text('q'+''.join(' %d:32'%i for i in range(10))+'\n',encoding='utf-8')
 paths.append(dict(label='conservative-overflow',gsp=gsp,queries=q,error='Query score bound does not fit uint32 shared scores'))
 gsp=root/'duplicate-ids.gsp';parent.write_gsp(gsp,[7,7],[[(0,1)],[(0,1)]],2);q=root/'duplicate-ids.tsv';q.write_text('q 0:1\n',encoding='utf-8')
 paths.append(dict(label='duplicate-ids',gsp=gsp,queries=q,error='Duplicate passage ID'))
 return paths
