"""Independent replay of the frozen query/process median and crossed bootstrap analysis."""
import argparse,csv,json,math,random,statistics
from pathlib import Path
import numpy as np

VARIANTS=['gpusparse-u64-mono-pl512','csr-full-byte-b16','csr-full-byte-b32']

def process(path):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    qids=list(dict.fromkeys(row['query_id'] for row in rows));assert len(qids)==500
    wanted=[(phase,str(r),q) for phase,n in [('warmup',3),('measured',20)] for r in range(n) for q in qids]
    assert [(r['phase'],r['round'],r['query_id']) for r in rows]==wanted
    values={q:[] for q in qids}
    for row in rows:
        value=float(row['query_wall_ms']);assert math.isfinite(value) and value>0
        if row['phase']=='measured':values[row['query_id']].append(value)
    assert {len(v) for v in values.values()}=={20}
    return qids,np.array([statistics.median(values[q]) for q in qids],dtype=np.float64)

def interval(logs):
    assert logs.shape==(3,500) and np.isfinite(logs).all()
    rng=random.Random(20261112);samples=np.empty(10000)
    for i in range(len(samples)):
        processes=rng.choices(range(3),k=3);queries=rng.choices(range(500),k=500)
        samples[i]=math.exp(float(logs[np.ix_(processes,queries)].mean()))
    return [float(v) for v in np.quantile(samples,[.025,.975],method='linear')]

def analyze(run):
    run=Path(run);values={};order=None
    for variant in VARIANTS:
        processes=[]
        for p in range(3):
            qids,times=process(run/'raw'/('timing-r%d-%s.csv'%(p,variant)))
            assert order is None or qids==order;order=qids;processes.append(times)
        values[variant]=np.stack(processes)
    server=json.loads((run/'analysis/contrasts.json').read_text());answer={}
    for variant in VARIANTS[1:]:
        logs=np.log(values[variant]/values[VARIANTS[0]])
        result=dict(csr_over_adapted_monolithic=math.exp(float(logs.mean())),
            conditional_crossed_bootstrap_95_interval=interval(logs),
            process_ratios=[math.exp(float(row.mean())) for row in logs],
            bootstrap_seed=20261112,bootstrap_resamples=10000)
        for key in ['csr_over_adapted_monolithic','conditional_crossed_bootstrap_95_interval','process_ratios']:
            assert np.allclose(result[key],server[variant][key],rtol=1e-12,atol=1e-15),(variant,key)
        answer[variant]=result
    return dict(success=True,independent_statistics=answer,server_statistics_match=True,
        descriptive_absolute_latency={v:dict(median_of_1500_query_process_medians_ms=float(np.median(x)),
        process_medians_ms=[float(np.median(row)) for row in x]) for v,x in values.items()},
        qualification='Complete-system comparison under primary uint64/canonical semantics on a guarded packed-selection domain. Fixed1M/k10/batch1/500calibration queries, three processes per variant on a shared RTX4070. Geometric mean of paired query/process ratios; conditional crossed-bootstrap intervals. Descriptive absolute medians are not the primary paired-ratio estimator. No held-out, original-native FP32, causal one-factor or service-tail inference.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args();result=analyze(a.run)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
