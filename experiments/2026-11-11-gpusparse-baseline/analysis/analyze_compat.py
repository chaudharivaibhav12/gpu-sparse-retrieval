"""Independent frozen native-path statistics from preserved per-query timings."""
import argparse,csv,json,math,random,statistics
from pathlib import Path
import numpy as np

def process(path):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    qids=list(dict.fromkeys(r['query_id'] for r in rows))
    assert len(qids)==500
    expected=[(phase,str(i),q) for phase,n in [('warmup',3),('measured',20)] for i in range(n) for q in qids]
    assert [(r['phase'],r['round'],r['query_id']) for r in rows]==expected
    values={q:[] for q in qids}
    for r in rows:
        latency=float(r['query_wall_ms']);assert math.isfinite(latency) and latency>0
        if r['phase']=='measured':values[r['query_id']].append(latency)
    assert all(len(v)==20 for v in values.values())
    return qids,np.array([statistics.median(values[q]) for q in qids],dtype=np.float64)

def interval(logs):
    assert logs.shape==(3,500) and np.isfinite(logs).all()
    rng=random.Random(20261111);boot=np.empty(10000,dtype=np.float64)
    for i in range(10000):
        ps=rng.choices(range(3),k=3);qs=rng.choices(range(500),k=500)
        boot[i]=math.exp(float(logs[np.ix_(ps,qs)].mean()))
    return [float(x) for x in np.quantile(boot,[.025,.975],method='linear')]

def analyze(run):
    run=Path(run);results={};absolute={};order=None
    for block in [128,512]:
        medians={}
        for variant in ['monolithic','chunked','compacted']:
            rows=[]
            for p in range(3):
                qids,times=process(run/'raw'/('timing-r%d-pl%d-%s.csv'%(p,block,variant)))
                if order is None:order=qids
                assert qids==order;rows.append(times)
            medians[variant]=np.stack(rows)
        results[str(block)]={};absolute[str(block)]={v:dict(median_of_1500_query_process_medians_ms=float(np.median(values)),process_medians_ms=[float(np.median(row)) for row in values]) for v,values in medians.items()}
        for variant in ['chunked','compacted']:
            logs=np.log(medians[variant]/medians['monolithic'])
            results[str(block)][variant]=dict(over_monolithic=math.exp(float(logs.mean())),conditional_crossed_bootstrap_95_interval=interval(logs),process_ratios=[math.exp(float(row.mean())) for row in logs],bootstrap_resamples=10000,bootstrap_seed=20261111)
    server=json.loads((run/'analysis/contrasts.json').read_text())
    for b,contrasts in results.items():
        for variant,value in contrasts.items():
            other=server[b][variant]
            for key in ['over_monolithic','conditional_crossed_bootstrap_95_interval','process_ratios']:
                assert np.allclose(value[key],other[key],rtol=1e-12,atol=1e-15),(b,variant,key)
    return dict(success=True,independent_statistics=results,descriptive_absolute_latency=absolute,server_statistics_match=True,qualification='BLOCK_PL512 primary;128 sensitivity. Within explicitly compatibility-patched FP32/native-tie treatment only. Geometric mean of paired ratios of20-round query medians; crossed process/query bootstrap,3processes/500calibration queries. No crossBLOCK_PL causality,primary-U64 speedup,held-out or service-tail claim.')

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args();result=analyze(a.run)
    with Path(a.output).open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
