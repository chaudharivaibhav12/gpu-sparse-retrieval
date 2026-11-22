"""Complete expected state for unchanged full-collection prunets, without CUDA.

Templates come from hash-bound accepted runtime snapshots, not subset gates.
Only explicitly enumerated per-run fields may change. Compiler/device changes
that alter occupancy or allocation must fail the gate and receive a new review.
"""
import hashlib,json
from pathlib import Path

def expected(repository,index,queries,metadata,query_rows,repeats,warmups,verify):
    repository=Path(repository)
    template=json.loads((Path(__file__).resolve().parents[1]/'configs/pruned-runtime-templates.json').read_text())
    for name,digest in template['source_sha256'].items():
        if hashlib.sha256((repository/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed; review and replace runtime model: '+name)
    width=metadata['range_size']
    if str(width) not in template['templates']:raise ValueError('Only frozen W256/W1024')
    state=template['templates'][str(width)]['runtime'];config=dict(state['config'])
    shape={'docs':'docs','vocab':'vocab','nnz':'nnz','ranges':'ranges','blocks':'blocks','index_posting_bytes':'posting_bytes','index_sha256':'body_sha256','source_gsp_sha256':'source_sha256'}
    if any(config[k]!=metadata[v] for k,v in shape.items()):raise ValueError('Different full index; new runtime model required')
    if not query_rows or len({q for q,_ in query_rows})!=len(query_rows):raise ValueError('Empty/duplicate query set')
    bound=0;max_terms=0
    for _,terms in query_rows:
        if len(terms)>128 or len({t for t,_ in terms})!=len(terms):raise ValueError('Invalid query terms')
        if any(not isinstance(t,int) or not isinstance(w,int) or not 0<=t<metadata['vocab'] or not 0<=w<=32 for t,w in terms):raise ValueError('Unsupported term/weight')
        score_bound=255*sum(w for _,w in terms)
        if score_bound>=2**24:raise ValueError('Outside common exact FP32 integer domain')
        bound=max(bound,score_bound);max_terms=max(max_terms,len(terms))
    if not isinstance(repeats,int) or repeats<1 or not isinstance(warmups,int) or warmups<0 or verify not in (0,1):raise ValueError('Invalid run settings')
    config.update(index_path=str(index),queries_path=str(queries),query_count=len(query_rows),max_query_terms=max_terms,max_conservative_score_bound=bound,repeats=repeats,warmup_rounds=warmups,verify_scores=verify)
    return {'config':config,'behavior':dict(state['behavior'])}
