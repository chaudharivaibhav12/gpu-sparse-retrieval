"""Canonical output checks precede any paired timing interpretation."""
import collections,csv,json,math,random,statistics
from pathlib import Path

CHECKS=['evaluated_score_mismatches','ordered_id_mismatches','result_score_mismatches','unsafe_pruned_documents','invalid_upper_bounds','invalid_schedule','threshold_mismatches']

def rows(path,delimiter=','):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter=delimiter))

def reference(path,k=None):
    path=Path(path)
    if path.suffix=='.json':
        data=json.loads(path.read_text(encoding='utf-8'))
        return {q:[(r+1,int(v[0]),int(v[1])) for r,v in enumerate(values[:k])] for q,values in data.items()}
    data={}
    for r in rows(path,'\t'):data.setdefault(r['query_id'],[]).append((int(r['rank']),int(r['original_id']),int(r['score'])))
    return data

def validate(csv_path,top_path,expected,B,mode,repeats,warmups,verify):
    samples=rows(csv_path);actual=collections.defaultdict(list);failures=[]
    for r in rows(top_path,'\t'):
        key=(r['phase'],int(r['round']),int(r['pruned']),r['query_id'])
        actual[key].append((int(r['rank']),int(r['original_id']),int(r['score'])))
    modes=[0,1] if mode==2 else [mode]
    wanted={(phase,r,p,q) for phase,n in [('warmup',warmups),('measured',repeats)] for r in range(n) for p in modes for q in expected}
    seen=set();thresholds=collections.defaultdict(set);qorder={q:i for i,q in enumerate(expected)}
    for r in samples:
        key=(r['phase'],int(r['round']),int(r['pruned']),r['query_id'])
        if key in seen:failures.append(['duplicate_sample',list(key)])
        seen.add(key)
        if key not in wanted:failures.append(['unexpected_sample',list(key)]);continue
        if actual.get(key,[])!=expected[key[-1]]:failures.append(['canonical_mismatch',list(key)])
        if any(r[c]!=('0' if verify else 'not_checked') for c in CHECKS):failures.append(['validation_counter',list(key)])
        blocks=(int(r['docs'])+B-1)//B
        seed=int(r['seed_blocks']);active=int(r['active_scored_blocks']);skipped=int(r['skipped_blocks'])
        if seed+active+skipped!=blocks or min(seed,active,skipped)<0:failures.append(['block_accounting',list(key)])
        if not key[2] and (seed or skipped or active!=blocks):failures.append(['control_schedule',list(key)])
        if key[2]:thresholds[key[-1]].add(int(r['seed_threshold']))
        order=int(r['order'])
        if mode==2 and (order not in [0,1] or key[2]!=(key[1]+qorder[key[-1]]+order)%2):failures.append(['paired_order',list(key)])
        if mode!=2 and order!=0:failures.append(['single_order',list(key)])
        if float(r['query_wall_ms'])<=0:failures.append(['invalid_latency',list(key)])
    if seen!=wanted:failures.append(['sample_membership',len(wanted-seen),len(seen-wanted)])
    sequence=[(phase,r,(r+qi+order)%2 if mode==2 else mode,q)
              for phase,n in [('warmup',warmups),('measured',repeats)] for r in range(n)
              for qi,q in enumerate(expected) for order in range(2 if mode==2 else 1)]
    observed=[(r['phase'],int(r['round']),int(r['pruned']),r['query_id']) for r in samples]
    if observed!=sequence:failures.append(['iteration_order'])
    if set(actual)-wanted:failures.append(['unexpected_topk_group'])
    if any(len(v)>1 for v in thresholds.values()):failures.append(['unstable_seed_threshold'])
    return {'success':not failures,'samples':len(samples),'expected_samples':len(wanted),'queries':len(expected),'block_size':B,'mode':mode,'repeats':repeats,'warmups':warmups,'failures':failures}

def paired_statistics(path):
    measured=[r for r in rows(path) if r['phase']=='measured'];groups=collections.defaultdict(list)
    for r in measured:groups[r['query_id'],int(r['pruned'])].append(float(r['query_wall_ms']))
    per_query=[]
    for q in sorted({r['query_id'] for r in measured}):
        a=groups[q,0];b=groups[q,1];assert len(a)==len(b)>0
        control=statistics.median(a);pruned=statistics.median(b)
        per_query.append({'query_id':q,'control_median_ms':control,'pruned_median_ms':pruned,'pruned_over_control':pruned/control})
    logs=[math.log(r['pruned_over_control']) for r in per_query];rng=random.Random(20261111)
    boot=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
    def quantile(v,p):
        x=(len(v)-1)*p;i=math.floor(x);j=math.ceil(x);return v[i]+(v[j]-v[i])*(x-i)
    ratio=math.exp(statistics.mean(logs))
    return {'primary':'geometric mean across query-wise pruned/control median latency ratios','pruned_over_control':ratio,'conditional_query_pair_bootstrap_95_interval':[quantile(boot,.025),quantile(boot,.975)],'bootstrap_resamples':10000,'bootstrap_seed':20261111,'per_query':per_query,'measured_samples':len(measured),'limitations':['25 repeated calibration queries on one shared GPU; conditional query-pair interval, not held-out or service-tail inference.','Both paths share a process with the pruning metadata resident; no control peak-memory claim can be derived from this paired process.','Diagnostic threshold transfer and file I/O are outside the query boundary; survivor-count transfer is included.']}
