"""Adapted GPUSparse monolithic traversal with actual U64 accumulation and canonical selection.

Derived from ashutoshuiuc/gpu-sparse revision92bb87b36f10e78ab39c81e9db2412dbb7a3b650,
src/triton_kernel.py SHA256ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0.
Original posting traversal preserved; products/accumulator and final selector explicitly changed.
This is not unmodified released GPUSparse. Batch1,k1..128; score bound<=2^31-1.
"""
import torch
import triton
import triton.language as tl
ACCUMULATOR_DTYPE=torch.uint64
KEY_DTYPE=torch.int64
PACK_THREADS=256
SCORE_BITS=31
ID_BITS=32

@triton.jit
def _fast_scatter_add_kernel(
    # Inverted index data
    doc_ids_ptr,
    doc_scores_ptr,
    offsets_ptr,
    lengths_ptr,
    # Query data
    query_term_ids_ptr,
    query_term_scores_ptr,
    # Output
    out_scores_ptr,
    # Dims
    num_docs,
    max_qterms: tl.constexpr,
    vocab_size,
    BLOCK_PL: tl.constexpr,  # Posting list chunk size
):
    """
    Simpler fused kernel: each program handles one (query, query_term) pair.
    Grid: [batch, max_qterms]

    Loads posting list for the query term and scatter-adds scores into the output.
    Uses tl.atomic_add for concurrent accumulation across terms.
    """
    q_idx = tl.program_id(0).to(tl.int64)
    t_pos = tl.program_id(1).to(tl.int64)

    # Load query term. Bound-check against vocab_size before using term_id as an
    # index: offsets_ptr and lengths_ptr are only vocab_size long, so an
    # out-of-range term_id reads garbage metadata and then issues unmasked wild
    # loads from doc_ids_ptr and wild atomics into out_scores_ptr. vocab_size was
    # already a kernel parameter but was previously unused.
    term_id = tl.load(query_term_ids_ptr + q_idx * max_qterms + t_pos)
    if term_id < 0 or term_id >= vocab_size:
        return

    q_score = tl.load(query_term_scores_ptr + q_idx * max_qterms + t_pos)

    # Load posting list metadata
    pl_offset = tl.load(offsets_ptr + term_id)
    pl_length = tl.load(lengths_ptr + term_id)

    if pl_length == 0:
        return

    # Process posting list in chunks
    n_chunks = (pl_length + BLOCK_PL - 1) // BLOCK_PL

    for chunk in range(n_chunks):
        chunk_start = pl_offset + chunk * BLOCK_PL
        offsets = tl.arange(0, BLOCK_PL)
        mask = offsets < (pl_length - chunk * BLOCK_PL)

        # Load doc IDs and scores
        pl_doc_ids = tl.load(doc_ids_ptr + chunk_start + offsets, mask=mask, other=-1)
        pl_scores = tl.load(doc_scores_ptr + chunk_start + offsets, mask=mask, other=0.0)

        # Compute contributions
        contribs = q_score.to(tl.uint64) * pl_scores.to(tl.uint64)

        # Scatter-add into output (int64 to avoid overflow at large batch * num_docs)
        out_offsets = q_idx * num_docs + pl_doc_ids.to(tl.int64)
        tl.atomic_add(out_scores_ptr + out_offsets, contribs, mask=mask & (pl_doc_ids >= 0))

@triton.jit
def _canonical_keys(scores, ranks, keys, n_docs, BLOCK:tl.constexpr):
    x=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    mask=x<n_docs
    score=tl.load(scores+x,mask=mask,other=0).to(tl.uint64)
    rank=tl.load(ranks+x,mask=mask,other=0).to(tl.uint64)
    key=(score<<32)|(tl.full((),4294967295,tl.uint64)-rank)
    tl.store(keys+x,key.to(tl.int64),mask=mask)

def canonical_monolithic(index,term_ids,term_scores,top_k,block_pl):
    assert term_ids.ndim==2 and term_ids.shape==term_scores.shape and term_ids.shape[0]==1
    assert 1<=top_k<=128 and 0<=index.num_docs<=2147483647
    assert block_pl>0 and not block_pl&(block_pl-1)
    n=index.num_docs;k=min(top_k,n)
    if n==0:
        return (torch.empty((1,0),device=index.device,dtype=torch.int64),torch.empty((1,0),device=index.device,dtype=torch.int64))
    scores=torch.zeros((1,n),device=index.device,dtype=ACCUMULATOR_DTYPE)
    _fast_scatter_add_kernel[(1,term_ids.shape[1])](index.doc_ids,index.scores,index.offsets,index.lengths,term_ids,term_scores,scores,n,term_ids.shape[1],index.vocab_size,BLOCK_PL=block_pl)
    keys=torch.empty((1,n),device=index.device,dtype=KEY_DTYPE)
    _canonical_keys[(triton.cdiv(n,PACK_THREADS),)](scores,index.id_ranks,keys,n,BLOCK=PACK_THREADS)
    selected,positions=torch.topk(keys,k=k,dim=1,largest=True,sorted=True)
    # Values fit31bits by the adapter's prequery domain guard; this is exact integer decoding.
    return selected>>ID_BITS,positions
