# Exact nonnegative-integer FP32 domain; no CUDA imports.
LIMIT=2**24
CONTRACT='GSP-CAP32-F32GUARDED-NUMERIC-TIES-v1'
ARITHMETIC='float32-guarded-exact-integer-sums'
def check_bound(bound):
    if type(bound) is not int or not 0<=bound<LIMIT:raise ValueError('FP32 exact-integer score bound must be below2^24')
    return bound
def query_bound(rows):
    if not rows:raise ValueError('At least one query required')
    seen=set();maximum=0
    for qid,terms in rows:
        if not qid or qid in seen:raise ValueError('Duplicate/empty query ID')
        seen.add(qid);used=set()
        if len(terms)>128:raise ValueError('More than128query terms')
        for term,weight in terms:
            if type(term) is not int or term<0 or term in used or type(weight) is not int or not 0<=weight<=32:raise ValueError('Expected unique cap32 nonnegative integer terms/weights')
            used.add(term)
        maximum=max(maximum,check_bound(255*sum(w for t,w in terms)))
    return maximum
