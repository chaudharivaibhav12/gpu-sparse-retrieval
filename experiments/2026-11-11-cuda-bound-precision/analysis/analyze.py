"""Canonical checks and frozen full/coarse/selective calibration contrasts."""
import argparse,collections,importlib.util,json,math,random,statistics
from pathlib import Path
_root=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('precision_parent_analysis',_root/'2026-11-11-cuda-fullbound-pruning/analysis/analyze.py')
parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(parent)
rows=parent.rows;reference=parent.reference

def validate(csv_path,top_path,expected,B,mode,repeats,warmups,verify,precision,vocab):
    result=parent.validate(csv_path,top_path,expected,B,mode,repeats,warmups,verify)
    failures=result['failures']
    for r in rows(csv_path):
        N=int(r['docs']);blocks=(N+B-1)//B;full=blocks*vocab if mode else 0;packed=(full+1)//2
        high=packed if precision else 0;low=packed if precision==2 else 0;flags=blocks*4 if mode and precision==2 else 0
        if int(r['precision'])!=precision:failures.append(['precision_setting',r['query_id']])
        for key,value in [('high_bytes',high),('low_bytes',low),('refinement_flags_bytes',flags),('bounds_bytes',high+low if precision else full)]:
            if int(r[key])!=value:failures.append(['storage_accounting',key,r['query_id']])
        count_bytes=(8 if precision==2 else 4) if int(r['pruned']) and blocks else 0
        if int(r['count_transfer_bytes'])!=count_bytes:failures.append(['count_transfer_boundary',r['query_id']])
        refined=int(r['refined_blocks'])
        if refined<0 or refined>blocks-int(r['seed_blocks']) or ((precision!=2 or not int(r['pruned'])) and refined):failures.append(['refinement_count',r['query_id']])
    result.update(success=not failures,precision=precision,vocab=vocab)
    return result

def schedule_check(csvs):
    """Cross-treatment invariant; internal VERIFY1 also checks every block flag."""
    schedules={}
    for precision,path in csvs.items():
        by_query={}
        for r in rows(path):
            if r['pruned']!='1':continue
            key=r['query_id'];value=tuple(int(r[x]) for x in ['seed_blocks','seed_docs','active_scored_blocks','skipped_blocks','seed_threshold','refined_blocks'])
            if key in by_query:assert by_query[key]==value,'Nondeterministic schedule'
            by_query[key]=value
        schedules[precision]=by_query
    assert set(schedules)=={0,1,2} and set(schedules[0])==set(schedules[1])==set(schedules[2])
    for q,a in schedules[0].items():
        coarse=schedules[1][q];selective=schedules[2][q]
        assert a[:5]==selective[:5],('Selective/full survivor mismatch',q)
        assert a[:2]==coarse[:2] and a[4]==coarse[4] and coarse[2]>=a[2] and coarse[3]<=a[3],('Coarse not a safe superset',q)
    return dict(success=True,queries=len(schedules[0]),rule='Selective/full schedules identical; coarse survivors a superset at the same real seed threshold.')

def _medians(path):
    groups=collections.defaultdict(list);ordered=[];seen=set()
    for r in rows(path):
        if r['phase']!='measured':continue
        q=r['query_id']
        if q not in seen:seen.add(q);ordered.append(q)
        groups[q,int(r['pruned'])].append(float(r['query_wall_ms']))
    assert len(ordered)==500 and len(groups)==1000 and {len(v) for v in groups.values()}=={20}
    return ordered,{q:(statistics.median(groups[q,0]),statistics.median(groups[q,1])) for q in ordered}

def contrasts(paths):
    """Three independent processes/treatment; direct latency primary, normalized sensitivity."""
    assert set(paths)=={0,1,2} and all(len(v)==3 for v in paths.values())
    assert len({str(Path(p).resolve()) for values in paths.values() for p in values})==9,'Expected nine distinct process CSVs'
    med={};queries=None
    for precision,files in paths.items():
        med[precision]=[]
        for path in files:
            assert {int(r['precision']) for r in rows(path)}=={precision},'Mislabeled precision CSV'
            q,m=_medians(path);assert queries is None or q==queries;queries=q;med[precision].append(m)
    answer={}
    for precision in [1,2]:
        direct=[];normalized=[]
        for process in range(3):
            direct.append([math.log(med[precision][process][q][1]/med[0][process][q][1]) for q in queries])
            normalized.append([math.log((med[precision][process][q][1]/med[precision][process][q][0])/(med[0][process][q][1]/med[0][process][q][0])) for q in queries])
        rng=random.Random(20261111);boot=[]
        for _ in range(10000):
            ps=rng.choices(range(3),k=3);qs=rng.choices(range(500),k=500)
            boot.append(math.exp(sum(direct[p][q] for p in ps for q in qs)/1500))
        boot.sort()
        def quantile(p):
            x=(len(boot)-1)*p;i=math.floor(x);return boot[i]+(boot[math.ceil(x)]-boot[i])*(x-i)
        answer[str(precision)]=dict(primary='Geometric mean of pruned latency relative to full-byte, equally weighted over500queries and3processes.',pruned_over_full=math.exp(statistics.mean(v for line in direct for v in line)),crossed_query_process_bootstrap_95_interval=[quantile(.025),quantile(.975)],normalized_control_sensitivity_ratio=math.exp(statistics.mean(v for line in normalized for v in line)),process_ratios=[math.exp(statistics.mean(v)) for v in direct],bootstrap_seed=20261111,bootstrap_resamples=10000,per_query=[dict(query_id=q,process_ratios=[math.exp(direct[p][i]) for p in range(3)]) for i,q in enumerate(queries)],limitations=['Separate processes/treatment with balanced fixed orders, not a within-process three-way randomization.','SharedGPU, uncontrolledclocks/cache; only3processes, repeated calibration; no held-out/service-tail or causal cross-host conclusion.'])
    return answer

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',nargs=3,required=True);p.add_argument('--coarse',nargs=3,required=True);p.add_argument('--selective',nargs=3,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=contrasts({0:a.full,1:a.coarse,2:a.selective})
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
