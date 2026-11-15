"""Untimed independent integer replay of saved oracle winners from original GSP rows."""
import csv,json,mmap,struct
from pathlib import Path
from expected import digest

def replay(gsp,queries,oracle,output):
 weights={}
 for line in Path(queries).read_text(encoding='utf-8').splitlines():
  words=line.split();weights[words[0]]=dict(tuple(map(int,word.split(':'))) for word in words[1:])
 groups={}
 with Path(oracle).open(encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f,delimiter='\t'):groups.setdefault(row['query_id'],[]).append((int(row['original_id']),int(row['score'])))
 assert set(groups)==set(weights)
 selected={d for rows in groups.values() for d,s in rows};documents={}
 with Path(gsp).open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
  assert data[:8]==b'GSPIDX01';n,v,nnz=struct.unpack_from('<IIQ',data,8)
  offsets=24+8*n;terms=offsets+8*(n+1);impacts=terms+4*nnz
  for position in range(n):
   identity=struct.unpack_from('<Q',data,24+8*position)[0]
   if identity not in selected:continue
   a,b=struct.unpack_from('<2Q',data,offsets+8*position)
   documents[identity]=[(struct.unpack_from('<I',data,terms+4*i)[0],data[impacts+i]) for i in range(a,b)]
 assert len(documents)==len(selected)
 count=0
 for q,rows in groups.items():
  assert rows==sorted(rows,key=lambda x:(-x[1],x[0]))
  for identity,score in rows:
   assert sum(impact*weights[q].get(term,0) for term,impact in documents[identity])==score,(q,identity)
   count+=1
 with Path(output).open('x',encoding='utf-8',newline='\n') as f:
  for identity,terms in sorted(documents.items()):f.write(json.dumps(dict(original_id=identity,postings=terms))+'\n')
 return dict(success=True,scores_checked=count,documents=len(documents),source_sha256=digest(gsp),queries_sha256=digest(queries),oracle_sha256=digest(oracle),rows_sha256=digest(output),scope='Returned oracle scores independently recomputed using unbounded Python integer arithmetic; does not prove omitted documents have lower scores. Complete top-k membership relies on the independent exhaustive CPU oracle.')
