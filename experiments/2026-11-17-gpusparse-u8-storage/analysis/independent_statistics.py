"""Independent standard-library replay of full-baseline CSV statistics.
Uses expanded Cartesian process/query bootstrap draws, not count-matrix weights.
"""
import csv,math,random,statistics
from pathlib import Path

def quantile(values,p):
 x=sorted(values);where=(len(x)-1)*p;lo=math.floor(where);hi=math.ceil(where)
 return x[lo]+(x[hi]-x[lo])*(where-lo)

def compute(files,qids,seed=2026111603,draws=10000):
 assert set(files)=={'range','control','compact'} and len(qids)==500 and len(set(qids))==500
 matrices={};summaries={}
 for label,paths in files.items():
  assert len(paths)==3;matrix=[]
  for path in paths:
   samples={qid:[] for qid in qids};seen=[]
   with Path(path).open(newline='') as f:
    for row in csv.DictReader(f):
     phase,r,q=row['phase'],int(row['round']),row['query_id'];seen.append((phase,r,q));x=float(row['query_wall_ms'])
     assert math.isfinite(x) and x>0 and int(row['docs'])==8841823 and int(row['k'])==10
     if phase=='measured':samples[q].append(x)
   assert seen==[(phase,r,q) for phase,count in [('warmup',3),('measured',20)] for r in range(count) for q in qids]
   assert all(len(v)==20 for v in samples.values());matrix.append([statistics.median(samples[q]) for q in qids])
  matrices[label]=matrix;flat=[x for row in matrix for x in row]
  summaries[label]={'median_query_process_median_ms':statistics.median(flat),'mean_query_process_median_ms':math.fsum(flat)/len(flat),'p95_query_process_median_ms':quantile(flat,.95),'process_medians_ms':[statistics.median(row) for row in matrix]}
 contrasts={}
 for i,(label,numerator,denominator) in enumerate([('compact_over_control','compact','control'),('range_over_compact','range','compact'),('range_over_control','range','control')]):
  logs=[[math.log(x/y) for x,y in zip(a,b)] for a,b in zip(matrices[numerator],matrices[denominator])]
  rng=random.Random(seed+i);values=[]
  for _ in range(draws):
   ps=rng.choices(range(3),k=3);qs=rng.choices(range(500),k=500)
   values.append(math.exp(math.fsum(logs[p][q] for p in ps for q in qs)/1500))
  ratio=math.exp(math.fsum(x for row in logs for x in row)/1500);lo,hi=quantile(values,.025),quantile(values,.975)
  contrasts[label]={'paired_latency_ratio':ratio,'conditional95':[lo,hi],'inverse_speedup':1/ratio,'inverse_speedup_conditional95':[1/hi,1/lo],'per_process_ratios':[math.exp(math.fsum(row)/500) for row in logs],'bootstrap_seed':seed+i,'bootstrap_draws':draws,'classification':numerator+'_lower_latency_in_this_calibration_scope' if hi<1 else numerator+'_higher_latency_in_this_calibration_scope' if lo>1 else 'inconclusive'}
 return {'summary':summaries,'comparisons':contrasts}
