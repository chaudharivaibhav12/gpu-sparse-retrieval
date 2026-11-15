"""Frozen paired sweep/confirmation statistics; standard library, no GPU imports."""
import csv,math,random,statistics
from pathlib import Path

def medians(path,qids):
 with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
 expected=[(phase,str(r),q) for phase,n in [('warmup',3),('measured',20)] for r in range(n) for q in qids]
 assert len(qids)==500 and [(r['phase'],r['round'],r['query_id']) for r in rows]==expected
 values={q:[] for q in qids};memory=set()
 for row in rows:
  value=float(row['query_wall_ms']);assert math.isfinite(value) and value>0;memory.add(int(row['device_allocation_bytes']))
  if row['phase']=='measured':values[row['query_id']].append(value)
 assert len(memory)==1 and {len(v) for v in values.values()}=={20}
 return [statistics.median(values[q]) for q in qids],memory.pop()

def quantile(ordered,p):
 x=(len(ordered)-1)*p;i=int(x);return ordered[i]+(ordered[math.ceil(x)]-ordered[i])*(x-i)

def contrast(a,b,seed):
 assert len(a)==len(b)==3 and all(len(x)==500 for x in a+b)
 logs=[[math.log(x/y) for x,y in zip(a[p],b[p])] for p in range(3)];rng=random.Random(seed);boot=[]
 for _ in range(10000):
  ps=rng.choices(range(3),k=3);qs=rng.choices(range(500),k=500);totals=[sum(row[q] for q in qs) for row in logs]
  boot.append(math.exp(sum(totals[p] for p in ps)/1500))
 boot.sort();return dict(ratio=math.exp(statistics.mean(x for row in logs for x in row)),conditional95=[quantile(boot,.025),quantile(boot,.975)],process_ratios=[math.exp(statistics.mean(row)) for row in logs],seed=seed,draws=10000)

def analyze(paths,qids,control='w1024-t256',seed=20261115):
 assert control in paths and all(len(v)==3 for v in paths.values());values={};memory={}
 for label,files in paths.items():
  parts=[medians(p,qids) for p in files];assert len({m for v,m in parts})==1;values[label]=[v for v,m in parts];memory[label]=parts[0][1]
 result={}
 for label in paths:
  flat=sorted(x for row in values[label] for x in row)
  result[label]=dict(**contrast(values[label],values[control],seed),median_ms=statistics.median(flat),mean_ms=statistics.mean(flat),p95_query_process_median_ms=quantile(flat,.95),device_allocation_bytes=memory[label])
 eligible=[label for label,x in result.items() if label!=control and x['ratio']<=.98 and x['conditional95'][1]<1]
 selected=control
 if eligible:
  best=min(result[label]['ratio'] for label in eligible);near=[label for label in eligible if result[label]['ratio']<=best*1.02]
  selected=min(near,key=lambda label:(memory[label],result[label]['ratio'],label))
 pareto=[label for label,x in result.items() if not any(y['ratio']<=x['ratio'] and y['device_allocation_bytes']<=x['device_allocation_bytes'] and (y['ratio']<x['ratio'] or y['device_allocation_bytes']<x['device_allocation_bytes']) for other,y in result.items() if other!=label)]
 return dict(cells=result,selected_for_confirmation=selected,control=control,pareto=pareto,qualification='Calibration selection among nine cells; conditional crossed query/process intervals do not correct for selecting the best. P95 is across query/process medians, not serving-tail latency. Memory is tracked application buffers, not physical peak. Fresh frozen confirmation required before promotion.')
