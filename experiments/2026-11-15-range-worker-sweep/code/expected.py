"""Independent input/shape resolver for the bounded exact range scorer; no CUDA imports."""
import hashlib,struct
from pathlib import Path
QUERY_ARGUMENT_BYTES=struct.calcsize('<I')+128*struct.calcsize('<II')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def rbp_metadata(path):
 path=Path(path)
 with path.open('rb') as f:
  assert f.read(8)==b'GSPRBP02','Only reviewed position format'
  n,v,w,r=struct.unpack('<4I',f.read(16));blocks,nnz,words,bits=struct.unpack('<4Q',f.read(32));source=f.read(32).hex()
  assert struct.unpack('<2I',f.read(8))==(1,0),'Uniform position encoding, no cap'
  assert w in [256,512,1024] and r==(n+w-1)//w and words==(bits+31)//32+1
  posting=4*(v+1)+19*blocks+4*words+nnz
  assert path.stat().st_size==96+8*n+posting+32
  f.seek(-32,2);body_hash=f.read(32).hex()
 # Verify the independent parser's checksum without loading the index into RAM.
 h=hashlib.sha256();remaining=path.stat().st_size-32
 with path.open('rb') as f:
  while remaining:
   block=f.read(min(1048576,remaining));assert block;h.update(block);remaining-=len(block)
 assert h.hexdigest()==body_hash
 return dict(docs=n,vocab=v,range_size=w,ranges=r,blocks=blocks,nnz=nnz,posting_bytes=posting,body_sha256=body_hash,source_sha256=source,file_sha256=digest(path))

def gsp_metadata(path):
 """Validate GSP rows and derive maximum multiplicity independently with SciPy."""
 import numpy as np
 from scipy.sparse import csr_matrix
 path=Path(path)
 with path.open('rb') as f:
  assert f.read(8)==b'GSPIDX01';n,v,nnz=struct.unpack('<IIQ',f.read(16));assert max(n,v,nnz)<2**31
  ids=np.fromfile(f,dtype='<u8',count=n);offsets=np.fromfile(f,dtype='<u8',count=n+1);terms=np.fromfile(f,dtype='<u4',count=nnz);impacts=np.fromfile(f,dtype='u1',count=nnz)
  assert len(ids)==n and len(offsets)==n+1 and len(terms)==len(impacts)==nnz
  assert offsets[0]==0 and offsets[-1]==nnz and np.all(offsets[1:]>=offsets[:-1]) and len(np.unique(ids))==n
  assert not nnz or int(terms.max())<v
  for _ in range(v):
   length=f.read(4);assert len(length)==4;size=struct.unpack('<I',length)[0];word=f.read(size);assert len(word)==size;word.decode('utf-8')
  assert not f.read(1)
 matrix=csr_matrix((impacts,terms.view(np.int32),offsets.view(np.int64)),shape=(n,v),copy=False);matrix.sort_indices()
 if not nnz:multiplicity=0
 elif nnz==1:multiplicity=1
 else:
  same=matrix.indices[1:]==matrix.indices[:-1];edges=matrix.indptr[1:-1];edges=edges[(edges>0)&(edges<nnz)];same[edges-1]=False
  multiplicity=1 if not same.any() else int(np.diff(np.r_[0,np.flatnonzero(~same)+1,nnz]).max())
 return dict(docs=n,vocab=v,nnz=nnz,multiplicity=multiplicity,sha256=digest(path))

def queries(path,vocab,limit,multiplicity):
 result=[];seen=set()
 for line in Path(path).read_text(encoding='utf-8').splitlines():
  fields=line.split();assert fields and fields[0] not in seen;seen.add(fields[0]);terms=[];used=set()
  for token in fields[1:]:
   t,w=map(int,token.split(':'));assert 0<=t<vocab and 0<=w<=32 and t not in used;used.add(t);terms.append((t,w))
  assert len(terms)<=128 and 255*sum(w for t,w in terms)*multiplicity<2**32-1
  result.append((fields[0],terms))
 return result[:limit]

def config(index_path,query_path,meta,gsp,query_rows,k,threads,repeats,warmups,verify,cpu):
 assert meta['source_sha256']==gsp['sha256'] and (meta['docs'],meta['vocab'],meta['nnz'])==(gsp['docs'],gsp['vocab'],gsp['nnz'])
 assert 0<=k<=1000 and threads in [64,128,256] and meta['range_size'] in [256,512,1024]
 slots=(meta['range_size']+threads-1)//threads;cap=min(k,32*slots);warps=threads//32
 return dict(mode='cuda-range-v2',docs=meta['docs'],query_count=len(query_rows),k=k,gpu_id=0,index_path=str(index_path),queries_path=str(query_path),variant=5,range_size=meta['range_size'],postings_cap=0,ranges=meta['ranges'],blocks=meta['blocks'],score_threads=threads,merge_threads=threads,max_supported_k=1000,max_query_terms=max((len(t) for q,t in query_rows),default=0),max_supported_terms=128,max_term_document_multiplicity=gsp['multiplicity'],slots_per_thread=slots,warp_list_capacity=cap,shared_bytes=warps*cap*8+meta['range_size']*4,index_posting_bytes=meta['posting_bytes'],tie_rank_bytes=meta['docs']*4,verify_scores=verify,warmup_rounds=warmups,repeats=repeats,cpu_affinity=str(cpu),index_sha256=meta['body_sha256'],source_gsp_sha256=gsp['sha256'],batch_size=1,vocab=meta['vocab'],nnz=meta['nnz'],score_accumulator_bits=32,returned_score_bits=64,score_limit_exclusive=2**32-1,max_conservative_score_bound=max((255*sum(w for t,w in terms)*gsp['multiplicity'] for q,terms in query_rows),default=0),max_slots_per_thread=128,query_argument_bytes=QUERY_ARGUMENT_BYTES)

def marks():
 return dict(contract='GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1',arithmetic='uint32-guarded-exact-widened-uint64',document_impacts='uint8-exact',query_policy='BMP-f32-ceil-cap32',tie_order='score_desc_original_numeric_id_asc',eligibility='all_documents',layout='range-split-bitpacked-positions-term-major',schedule='exhaustive-all-ranges',ranges='boundaries-from-index;uniform-or-postings-capped',decode='combined-work-list;direct-positions;no-decode-barriers',accumulator='uint32-shared-gated-below-2^32-1-widened-to-uint64',selection='thread-sorted-slots;warp-shuffle-select;one-barrier-rank-scatter;merge-in-last-finishing-block-threadfence-atomic-ticket',tie_keys='per-range-original-id-rank-uint16',query_transfer='kernel-argument-by-value',result_transfer='topk-only')

def allocation(meta,k,threads):
 n=meta['docs'];r=meta['ranges'];warps=threads//32
 return meta['posting_bytes']+n*12+r*2+(r+2)*4+(r*k+warps*k+k)*24
