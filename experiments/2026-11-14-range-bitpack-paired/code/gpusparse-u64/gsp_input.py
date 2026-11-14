"""GSP/TSV input adapter and independent integer-domain checks; no CUDA imports."""
import hashlib,struct
from pathlib import Path
import numpy as np
import scipy.sparse as sp

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def load_index(path):
 path=Path(path)
 with path.open('rb') as f:
  if f.read(8)!=b'GSPIDX01':raise ValueError('Bad GSP magic')
  header=f.read(16)
  if len(header)!=16:raise ValueError('Truncated GSP header')
  N,V,nnz=struct.unpack('<IIQ',header)
  if N>np.iinfo(np.int32).max or V>np.iinfo(np.int32).max:raise ValueError('Native index exceeds int32 domain')
  if path.stat().st_size<24+8*N+8*(N+1)+5*nnz+4*V:raise ValueError('Truncated GSP payload')
  ids=np.fromfile(f,dtype='<u8',count=N);offsets=np.fromfile(f,dtype='<u8',count=N+1)
  terms=np.fromfile(f,dtype='<u4',count=nnz);impacts=np.fromfile(f,dtype='u1',count=nnz)
  for _ in range(V):
   raw=f.read(4)
   if len(raw)!=4:raise ValueError('Truncated vocabulary length')
   length=struct.unpack('<I',raw)[0];word=f.read(length)
   if len(word)!=length:raise ValueError('Truncated vocabulary')
   word.decode('utf-8')
  if f.read(1):raise ValueError('Unexpected GSP tail')
 if len(ids)!=N or len(offsets)!=N+1 or len(terms)!=nnz or len(impacts)!=nnz:raise ValueError('Truncated arrays')
 if offsets[0]!=0 or offsets[-1]!=nnz or np.any(offsets[1:]<offsets[:-1]):raise ValueError('Bad CSR offsets')
 if len(np.unique(ids))!=N:raise ValueError('Duplicate original numeric ID')
 if nnz and int(terms.max())>=V:raise ValueError('Term outside vocabulary')
 csr=sp.csr_matrix((impacts,terms.astype(np.int32),offsets.astype(np.int64)),shape=(N,V),copy=False)
 csr.sort_indices()
 if nnz>1:
  duplicate=csr.indices[1:]==csr.indices[:-1];edges=csr.indptr[1:-1];edges=edges[(edges>0)&(edges<nnz)]
  duplicate[edges-1]=False
  if duplicate.any():raise ValueError('Duplicate term in a passage')
 csc=csr.tocsc();csc.sort_indices()
 if np.any(np.diff(csc.indptr)>np.iinfo(np.int32).max):raise ValueError('Posting length exceeds int32')
 return dict(ids=ids,csr=csr,csc=csc,N=N,V=V,nnz=nnz)

def read_queries(path,V,limit):
 queries=[];seen=set()
 for line in Path(path).read_text(encoding='utf-8').splitlines():
  fields=line.split()
  if not fields or fields[0] in seen:raise ValueError('Empty/duplicate query ID')
  seen.add(fields[0]);terms=[];used=set()
  for token in fields[1:]:
   pieces=token.split(':')
   if len(pieces)!=2:raise ValueError('Malformed query term')
   term,weight=map(int,pieces)
   if term in used or not 0<=term<V or not 0<=weight<=32:raise ValueError('Query outside unique cap32 domain')
   used.add(term);terms.append((term,weight))
  if 255*sum(w for _,w in terms)>2**31-1:raise ValueError('Canonical packed-selection score bound exceeds2^31-1')
  queries.append((fields[0],terms))
 if limit<0:raise ValueError('Negative query limit')
 return queries[:limit]

def result_score(index,doc,terms):
 if not 0<=doc<index['N']:raise ValueError('Returned position outside index')
 row=index['csr'].getrow(doc);weights=dict(terms)
 return sum(int(value)*weights.get(int(term),0) for term,value in zip(row.indices,row.data))

def padded_query(terms,width):
 if width<max(1,len(terms)):raise ValueError('Query width truncates terms')
 ids=np.full((1,width),-1,dtype=np.int32);weights=np.zeros((1,width),dtype=np.float32)
 for i,(term,weight) in enumerate(terms):ids[0,i]=term;weights[0,i]=weight
 return ids,weights
