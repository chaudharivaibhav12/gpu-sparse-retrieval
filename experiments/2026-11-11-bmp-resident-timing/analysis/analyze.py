"""Validate every sample/output, then compute the preregistered paired statistics."""
import argparse,collections,csv,json,math,random,statistics
from pathlib import Path


def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))


def analyze(samples_path,results_path,oracle_path,warmups,repeats):
    oracle=collections.defaultdict(list)
    for row in read(oracle_path):
        oracle[row['query_id']].append((int(row['rank']),int(row['original_id']),int(row['score'])))
    assert oracle and all(len(v)==10 and [r[0] for r in v]==list(range(1,11)) for v in oracle.values())
    samples=read(samples_path)
    outputs=collections.defaultdict(list)
    for row in read(results_path):
        key=(row['phase'],int(row['round']),row['layout'],row['query_id'])
        outputs[key].append((int(row['rank']),int(row['document_id']),int(row['score'])))
    expected={(p,r,l,q) for p,n in [('warmup',warmups),('measured',repeats)]
              for r in range(n) for l in ['raw','compressed'] for q in oracle}
    failures=[]
    seen=set(); measured=collections.defaultdict(list); positions={}
    for row in samples:
        key=(row['phase'],int(row['round']),row['layout'],row['query_id'])
        if key in seen:failures.append(['duplicate_sample',list(key)])
        seen.add(key)
        if key not in expected:failures.append(['unexpected_sample',list(key)])
        qi=int(row['query_index']); order=int(row['order']);ns=int(row['elapsed_ns'])
        positions.setdefault(row['query_id'],set()).add(qi)
        expected_layout='compressed' if (int(row['round'])+qi+order)%2 else 'raw'
        if order not in (0,1) or row['layout']!=expected_layout or ns<=0:failures.append(['invalid_order_or_time',list(key)])
        if outputs.get(key)!=oracle.get(row['query_id']):failures.append(['top10_mismatch',list(key)])
        if row['phase']=='measured':measured[(row['layout'],row['query_id'])].append(ns)
    if seen!=expected:failures.append(['sample_membership',len(expected-seen),len(seen-expected)])
    if set(outputs)!=expected:failures.append(['output_membership',len(expected-set(outputs)),len(set(outputs)-expected)])
    if positions!={q:{i} for i,q in enumerate(oracle)}:
        failures.append(['query_order_inconsistent'])
    validation={'success':not failures,'queries':len(oracle),'warmup_rounds':warmups,'measured_rounds':repeats,
                'samples':len(samples),'expected_samples':len(expected),'failures':failures}
    if failures:return {'validation':validation,'timing':None}
    per_query=[]
    for q in sorted(oracle,key=lambda x:next(iter(positions[x]))):
        raw=measured['raw',q];compressed=measured['compressed',q]
        assert len(raw)==len(compressed)==repeats
        rm=statistics.median(raw);cm=statistics.median(compressed)
        per_query.append({'query_id':q,'raw_median_ms':rm/1e6,'compressed_median_ms':cm/1e6,'compressed_over_raw':cm/rm})
    logs=[math.log(x['compressed_over_raw']) for x in per_query]
    estimate=math.exp(statistics.mean(logs))
    rng=random.Random(20261111)
    boot=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
    def percentile(v,p):
        position=(len(v)-1)*p;low=math.floor(position);high=math.ceil(position)
        return v[low]+(v[high]-v[low])*(position-low)
    timing={'primary':'geometric mean of query-wise compressed/raw median ratios','compressed_over_raw':estimate,
            'bootstrap_95_interval':[percentile(boot,.025),percentile(boot,.975)],'bootstrap_seed':20261111,
            'bootstrap_unit':'query pair; conditional on the 25 calibration queries and this execution',
            'per_query':per_query,'layouts':{}}
    for layout in ['raw','compressed']:
        v=sorted(ns/1e6 for (l,q),values in measured.items() if l==layout for ns in values)
        timing['layouts'][layout]={'observations':len(v),'pooled_median_ms':statistics.median(v),'pooled_p95_ms':percentile(v,.95)}
    return {'validation':validation,'timing':timing,'limitations':[
        '25 repeated calibration queries on one shared CPU host; not held-out/general-population or service-tail inference.',
        'Both indexes resident; repeated-cache workload; resident context and file I/O excluded.',
        'Boundary includes cursor preparation through ordered TREC string construction; not GPU or load-inclusive latency.',
        'Affinity and round-level CPU snapshots do not guarantee exclusive cores or fixed frequency.'
    ]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--results',required=True)
    p.add_argument('--oracle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--warmups',type=int,default=3);p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args();result=analyze(a.samples,a.results,a.oracle,a.warmups,a.repeats)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['validation']))
    if not result['validation']['success']:raise SystemExit('Output/sample validation failed; timing interpretation blocked')
