import copy,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
R=Path(__file__).resolve().parents[3]
B=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('pruned_expected',B/'code/pruned_expected.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
t=json.loads((B/'configs/pruned-runtime-templates.json').read_text())
rows=[]
for line in (R/'results/calibration-500.tsv').read_text().splitlines():
 a=line.split();rows.append((a[0],[tuple(map(int,x.split(':'))) for x in a[1:]]))
count=0
for width in [256,1024]:
 actual=t['templates'][str(width)]['runtime'];c=actual['config']
 meta={v:c[k] for k,v in {'docs':'docs','vocab':'vocab','nnz':'nnz','ranges':'ranges','blocks':'blocks','index_posting_bytes':'posting_bytes','index_sha256':'body_sha256','source_gsp_sha256':'source_sha256'}.items()};meta['range_size']=width
 assert m.expected(R,c['index_path'],c['queries_path'],meta,rows,20,3,0)==actual;count+=1
 cases=[dict(query_rows=rows+[rows[0]]),dict(query_rows=[]),dict(query_rows=[('bad',[(0,33)])]),dict(query_rows=[('bad',[(0,1),(0,2)])]),dict(query_rows=[('bad',[(i,1) for i in range(129)])]),dict(query_rows=[('bad',[(meta['vocab'],1)])]),dict(repeats=0),dict(warmups=-1),dict(verify=2),dict(metadata=dict(meta,nnz=meta['nnz']+1)),dict(metadata=dict(meta,range_size=512))]
 for change in cases:
  args=dict(repository=R,index=c['index_path'],queries=c['queries_path'],metadata=meta,query_rows=rows,repeats=20,warmups=3,verify=0);args.update(change)
  try:m.expected(**args)
  except ValueError:count+=1
  else:raise AssertionError(change)
print(json.dumps({'checks':count,'full_state_roundtrips':2,'negative_rejections':count-2,'status':'PASS'}))
