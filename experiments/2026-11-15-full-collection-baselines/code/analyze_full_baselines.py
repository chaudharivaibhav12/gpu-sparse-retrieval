"""Frozen paired query/process estimator; no serving-tail or held-out claim."""
import csv,json,math,random
from pathlib import Path
import numpy as np

def contrast(log_ratios,seed,draws=10000):
    x=np.asarray(log_ratios,dtype=np.float64)
    if x.shape!=(3,500) or not np.isfinite(x).all():raise ValueError('paired data shape/domain')
    rng=random.Random(seed);samples=np.empty(draws,dtype=np.float64)
    for i in range(draws):
        p=np.bincount(rng.choices(range(3),k=3),minlength=3)
        q=np.bincount(rng.choices(range(500),k=500),minlength=500)
        samples[i]=math.exp(float(p@x@q)/1500)
    ratio=math.exp(float(x.mean()));lo,hi=map(float,np.quantile(samples,[.025,.975],method='linear'))
    return dict(range_over_baseline_ratio=ratio,conditional95=[lo,hi],inverse_speedup=1/ratio,inverse_speedup_conditional95=[1/hi,1/lo],per_process_ratios=list(map(float,np.exp(x.mean(axis=1)))),bootstrap_seed=seed,bootstrap_draws=draws,classification='range_lower_latency_in_this_calibration_scope' if hi<1 else 'range_higher_latency_in_this_calibration_scope' if lo>1 else 'inconclusive')

def analyze(entries,qids,seed):
    if len(entries)!=9 or len(qids)!=500 or len(set(qids))!=500:raise ValueError('fixed complete matrix required')
    matrices={};summary={}
    for label in ['range','mono','csr']:
        selected=sorted((e for e in entries if e['label']==label),key=lambda e:e['repetition'])
        if [e['repetition'] for e in selected]!=[0,1,2] or not all(e['eligible'] for e in selected):raise ValueError('process repetition/eligibility')
        rows=[];memory=[]
        for entry in selected:
            values={qid:{} for qid in qids}
            for row in entry['latencies']:
                if row['query_id'] not in values or not 0<=row['round']<20:raise ValueError('unexpected measured sample')
                if row['round'] in values[row['query_id']]:raise ValueError('duplicate measured query/round')
                values[row['query_id']][row['round']]=row['query_wall_ms']
            if any(set(v)!=set(range(20)) or not all(math.isfinite(x) and x>0 for x in v.values()) for v in values.values()):raise ValueError('missing/invalid repeats')
            rows.append([float(np.median(list(values[q].values()))) for q in qids])
            item={'sampled_own_peak_mib':entry['monitor']['sampled_own_peak_mib'],'sampled_board_peak_mib':entry['monitor']['sampled_board_peak_mib']}
            with Path(entry['csv']).open() as f:first=next(csv.DictReader(f))
            if label in ['range','csr']:item['tracked_application_bytes']=int(first['device_allocation_bytes'])
            else:
                native=json.loads((Path(entry['csv']).parent/'native-validation.json').read_text());item['torch_allocator_peak_after_setup_bytes']=native['torch_allocator_peak_bytes'];item['torch_reserved_peak_after_setup_bytes']=native['torch_allocator_reserved_peak_bytes']
            memory.append(item)
        matrix=np.asarray(rows,dtype=np.float64);matrices[label]=matrix
        summary[label]=dict(median_query_process_median_ms=float(np.median(matrix)),mean_query_process_median_ms=float(np.mean(matrix)),p95_query_process_median_ms=float(np.quantile(matrix,.95,method='linear')),process_medians_ms=list(map(float,np.median(matrix,axis=1))),processes=3,queries=500,measured_rounds=20,memory=memory)
    comparisons={label:contrast(np.log(matrices['range']/matrices[label]),seed+i) for i,label in enumerate(['mono','csr'])}
    return dict(full_passages=8841823,k=10,batch_size=1,summary=summary,comparisons=comparisons,qualification='Fixed500calibration queries and3processes per implementation; paired geometric ratios of20-round query/process medians. Crossed bootstrap intervals conditional on this set/session; no held-out/general-input/serving-tail inference. P95 describes query/process medians, not request latency tail. Sampled GPU peaks, C++ tracked allocations and Torch allocator peaks have different scopes; no claim that any is an exact physical peak.')
