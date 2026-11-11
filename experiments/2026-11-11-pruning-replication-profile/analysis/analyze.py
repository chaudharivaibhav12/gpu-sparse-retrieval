"""Compute frozen replication statistics from already-validated raw CSVs."""
import argparse,collections,csv,json,math,random,statistics
from pathlib import Path
def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def aggregate(paths,first25):
    matrix=[];per=[];queries=None
    for path in paths:
        groups=collections.defaultdict(list)
        for r in rows(path):
            if r['phase']=='measured':groups[r['query_id'],int(r['pruned'])].append(float(r['query_wall_ms']))
        qs=list(dict.fromkeys(r['query_id'] for r in rows(path)))
        assert len(qs)==500 and (queries is None or qs==queries);queries=qs
        logs=[math.log(statistics.median(groups[q,1])/statistics.median(groups[q,0])) for q in queries];matrix.append(logs)
        per.append(dict(csv=str(path),all500_ratio=math.exp(statistics.mean(logs)),historical25_ratio=math.exp(statistics.mean(v for q,v in zip(queries,logs) if q in first25)),queries_faster=sum(v<0 for v in logs)))
    rng=random.Random(20261111);boot=[]
    for _ in range(10000):
        ps=rng.choices(range(len(matrix)),k=len(matrix));qs=rng.choices(range(500),k=500)
        boot.append(math.exp(sum(matrix[p][q] for p in ps for q in qs)/(len(ps)*len(qs))))
    boot.sort()
    def quantile(p):
        x=(len(boot)-1)*p;i=math.floor(x);return boot[i]+(boot[math.ceil(x)]-boot[i])*(x-i)
    return dict(primary='Geometric mean of query ratios across fixed processes, without treating repeated rounds as independent queries.',pruned_over_control=math.exp(statistics.mean(v for line in matrix for v in line)),crossed_process_query_bootstrap_95_interval=[quantile(.025),quantile(.975)],bootstrap_seed=20261111,bootstrap_resamples=10000,processes=per,per_query=[dict(query_id=q,ratios=[math.exp(row[i]) for row in matrix]) for i,q in enumerate(queries)],limitations=['Only three processes, one shared GPU and repeated calibration inputs; conditional interval, not held-out or service-tail inference.'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',nargs=3);p.add_argument('--first25-reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert len({str(Path(q).resolve()) for q in a.csv})==3,'Use three distinct process CSVs with preserved run receipts'
    with Path(a.first25_reference).open(newline='',encoding='utf-8') as f:first25={r['query_id'] for r in csv.DictReader(f,delimiter='\t')}
    assert len(first25)==25
    for path in a.csv:
        counts=collections.Counter((r['query_id'],int(r['pruned'])) for r in rows(path) if r['phase']=='measured')
        assert len(counts)==1000 and set(counts.values())=={20},'Expected500queries x2paths x20measured rounds'
    result=aggregate(a.csv,first25);out=Path(a.output)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
