"""Independent NumPy/count-weighted implementation of the frozen range analysis."""
import csv,math,random
from pathlib import Path
import numpy as np

def contrast(log_ratios,seed):
 x=np.asarray(log_ratios,dtype=np.float64);assert x.shape==(3,500) and np.isfinite(x).all()
 rng=random.Random(seed);draws=np.empty(10000,dtype=np.float64)
 for i in range(len(draws)):
  # Multiplicity weights implement the Cartesian product of process/query draws.
  process_counts=np.bincount(rng.choices(range(3),k=3),minlength=3)
  query_counts=np.bincount(rng.choices(range(500),k=500),minlength=500)
  draws[i]=np.exp((process_counts@x@query_counts)/1500)
 limits=np.percentile(draws,[2.5,97.5],method='linear')
 return dict(ratio=float(np.exp(x.mean())),conditional95=list(map(float,limits)),process_ratios=list(map(float,np.exp(x.mean(axis=1)))),seed=seed,draws=10000)

def analyze(paths,qids,control='w1024-t256',seed=20261115):
 assert len(qids)==500 and len(set(qids))==500 and control in paths
 matrices={};memory={}
 for label,files in paths.items():
  assert len(files)==3;processes=[];allocations=set()
  for file in files:
   measurements={q:[] for q in qids}
   with Path(file).open(newline='') as f:
    for row in csv.DictReader(f):
     allocations.add(int(row['device_allocation_bytes']))
     if row['phase']=='measured':measurements[row['query_id']].append(float(row['query_wall_ms']))
   assert all(len(values)==20 for values in measurements.values())
   samples=np.asarray([measurements[q] for q in qids],dtype=np.float64);assert np.isfinite(samples).all() and (samples>0).all()
   processes.append(np.median(samples,axis=1))
  assert len(allocations)==1;memory[label]=allocations.pop();matrices[label]=np.asarray(processes)
 cells={}
 for label,matrix in matrices.items():
  cells[label]=dict(**contrast(np.log(matrix/matrices[control]),seed),median_ms=float(np.median(matrix)),mean_ms=float(matrix.mean()),p95_query_process_median_ms=float(np.percentile(matrix,95,method='linear')),device_allocation_bytes=memory[label])
 eligible={label for label,value in cells.items() if label!=control and value['ratio']<=.98 and value['conditional95'][1]<1}
 selected=control
 if eligible:
  best=min(cells[label]['ratio'] for label in eligible)
  near=[label for label in eligible if cells[label]['ratio']<=best*1.02]
  selected=sorted(near,key=lambda label:(memory[label],cells[label]['ratio'],label))[0]
 pareto=[]
 for label,value in cells.items():
  dominated=False
  for other,alternative in cells.items():
   if other==label:continue
   no_worse=alternative['ratio']<=value['ratio'] and memory[other]<=memory[label]
   better=alternative['ratio']<value['ratio'] or memory[other]<memory[label]
   if no_worse and better:dominated=True
  if not dominated:pareto.append(label)
 return dict(cells=cells,selected_for_confirmation=selected,control=control,pareto=pareto)
