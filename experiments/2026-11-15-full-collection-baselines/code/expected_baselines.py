"""Resolve audited baseline runtime schemas without importing CUDA packages."""
import importlib.util
from pathlib import Path

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def inherited(repository):
    repo=Path(repository)
    return (module('full_range_expected',repo/'experiments/2026-11-15-range-worker-sweep/code/expected.py'),module('full_mono_expected',repo/'experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64/expected_config.py'))

def csr(index,query,gsp,rows,k,rounds,warmups,verify,cpu,primary):
    n=gsp['docs'];blocks=(n+31)//32;groups=(n+255)//256
    levels=0;count=n
    if n and k:
        while True:
            levels+=1;g=(count+255)//256
            if g==1:break
            count=g*k
    scratch=2*groups*k*24
    cfg=dict(mode='cuda-fullbound-pruning',docs=n,query_count=len(rows),k=k,gpu_id=0,index_path=str(index),queries_path=str(query),batch_size=1,block_size=32,execution_mode=0,score_threads=32,selection_threads=256,bound_threads=256,compaction_threads=256,max_supported_k=128,selection_levels=levels,scratch_bytes=scratch,seed_selection_scratch_bytes=0,bounds_bytes=0,seed_budget_blocks=16,pruning_seed_blocks=0,pruning_seed_docs=0,max_query_terms=max((len(t) for q,t in rows),default=0),verify_scores=verify,warmup_rounds=warmups,repeats=rounds,cpu_affinity=str(cpu))
    behavior=dict(primary,selection='hierarchical-local-topk',layout='csr-document-blocks',bounds='term-major-u8-max-u64-sum',seed='first-16-identity-blocks',threshold='distinct-seed-kth-or-zero',skip_rule='strictly-below-threshold',compaction='stable-three-pass-prefix',skipped_scores='zeroed-before-each-query',paired_order='alternating-round-plus-query')
    allocation=24*n+8+5*gsp['nnz']+4*gsp['vocab']+4*blocks+scratch
    return cfg,behavior,allocation
