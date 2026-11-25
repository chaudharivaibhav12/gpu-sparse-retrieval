# Pinned upstream FP32 traversal plus unchanged canonical key-selection algorithm.
# The original source is hash-checked and imported before any query execution.
# The adapter's all-file integer-domain guard is required; this is a labelled adaptation.
import hashlib,importlib.util,sys
from pathlib import Path
import torch,triton
from u64_monolithic import _canonical_keys,KEY_DTYPE,PACK_THREADS,SCORE_BITS,ID_BITS
ACCUMULATOR_DTYPE=torch.float32
NATIVE_SHA='ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0'
_native_scatter=None

def load_native(path):
    global _native_scatter
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=NATIVE_SHA:raise ValueError('Pinned native scorer source changed')
    spec=importlib.util.spec_from_file_location('guarded_original_gp_scoring',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    _native_scatter=module._fast_scatter_add_kernel

def canonical_monolithic(index,term_ids,term_scores,top_k,block_pl):
    assert term_ids.ndim==2 and term_ids.shape==term_scores.shape and term_ids.shape[0]==1
    assert 1<=top_k<=1000 and 0<=index.num_docs<=2147483647
    assert block_pl>0 and not block_pl&(block_pl-1)
    n=index.num_docs;k=min(top_k,n)
    if n==0:
        return (torch.empty((1,0),device=index.device,dtype=torch.int64),torch.empty((1,0),device=index.device,dtype=torch.int64))
    scores=torch.zeros((1,n),device=index.device,dtype=ACCUMULATOR_DTYPE)
    _native_scatter[(1,term_ids.shape[1])](index.doc_ids,index.scores,index.offsets,index.lengths,term_ids,term_scores,scores,n,term_ids.shape[1],index.vocab_size,BLOCK_PL=block_pl)
    keys=torch.empty((1,n),device=index.device,dtype=KEY_DTYPE)
    _canonical_keys[(triton.cdiv(n,PACK_THREADS),)](scores,index.id_ranks,keys,n,BLOCK=PACK_THREADS)
    selected,positions=torch.topk(keys,k=k,dim=1,largest=True,sorted=True)
    # Values fit31bits by the adapter's prequery domain guard; this is exact integer decoding.
    return selected>>ID_BITS,positions
