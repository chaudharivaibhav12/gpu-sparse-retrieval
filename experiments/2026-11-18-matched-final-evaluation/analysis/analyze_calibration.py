"""Independent raw replay and preregistered crossed-bootstrap calibration analysis.

No controller-produced latency vectors are used. Standard library only. Ratios
are A/B query time; inverse ratios describe B/A speedup. Intervals are conditional
on the three observed process blocks, this shared host and calibration population.
"""
import argparse,csv,decimal,hashlib,json,math,random,statistics
from pathlib import Path
SYSTEMS=['prunets256','prunets1024','range','compact','bmp']
CONTRASTS=[('prunets256','compact','primary'),('prunets256','bmp','primary'),('prunets1024','compact','secondary'),('prunets1024','bmp','secondary'),('prunets256','range','secondary'),('prunets1024','range','secondary')]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def event(root,identity):
    value=read(root/'research/events'/(identity+'.json'));expected=value.pop('sha256')
    assert hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==expected
    return value
def quantile(data,p):
    ordered=sorted(data);at=(len(ordered)-1)*p;lo=math.floor(at);hi=math.ceil(at)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)
def crossed_draw(logs,rng):
    processes=len(logs);queries=len(logs[0]);ps=[rng.randrange(processes) for _ in range(processes)];qs=[rng.randrange(queries) for _ in range(queries)]
    return math.exp(sum(sum(logs[p][q] for q in qs) for p in ps)/(processes*queries))
def contrast(left,right,seed=20261118,draws=10000):
    assert len(left)==len(right)==3 and all(len(row)==len(left[0]) for row in left+right)
    logs=[[math.log(a/b) for a,b in zip(x,y)] for x,y in zip(left,right)]
    assert all(math.isfinite(v) for row in logs for v in row)
    point=math.exp(sum(map(sum,logs))/(3*len(logs[0])));rng=random.Random(seed);samples=[crossed_draw(logs,rng) for _ in range(draws)];interval=[quantile(samples,.025),quantile(samples,.975)]
    return dict(query_time_ratio_a_over_b=point,conditional95=interval,inverse_ratio=1/point,inverse_conditional95=[1/interval[1],1/interval[0]],pairs=3*len(logs[0]),draws=draws,seed=seed,rng='Python random.Random MT19937; crossed process and query draws with replacement, shared query sample across drawn processes')
def replay_monitor(directory,label):
    summary=read(directory/'monitor-summary.json');identity=read(directory/'process-identity.json');group=identity['runner_pid'];assert group==identity['expected_process_group']==summary['run_process_group']
    frames=[json.loads(line) for line in (directory/'monitor.jsonl').read_text().splitlines()];assert len(frames)==summary['samples'] and len(frames)>1
    foreign=set();owned_seen=False;peak=0;cpu=[];clocks=[];temperatures=[];last=None
    for f in frames:
        assert 'error' not in f and f['run_process_group']==group
        assert [(r['pid'],r['used_mib']) for r in f['context_ownership']]==[tuple(x) for x in f['gpu']['contexts']]
        own=0
        for r in f['context_ownership']:
            assert r['owned']==(r['pgid']==group)
            if r['owned']:owned_seen=True;own+=r['used_mib']
            else:foreign.add(r['pid'])
        peak=max(peak,own);assert f['own_swap_kib']<=1024**2
        if last:
            other=max(0.0,((f['machine_busy_ticks']-last['machine_busy_ticks'])-(f['own_group_cpu_ticks']-last['own_group_cpu_ticks']))/f['ticks_per_second']/(f['sample_monotonic']-last['sample_monotonic']))
            assert math.isclose(other,f['other_cpu_cores'],rel_tol=1e-12,abs_tol=1e-12);cpu.append(other)
        last=f
        for key,out in [('sm_clock_mhz',clocks),('temperature_c',temperatures)]:
            try:out.append(float(f['gpu']['telemetry'][key]))
            except ValueError:pass
    assert not summary['errors'] and sorted(foreign)==summary['foreign_pids']==[]
    assert summary['own_gpu_context_seen']==owned_seen and peak==summary['sampled_own_peak_mib']
    assert label=='bmp' or owned_seen
    assert math.isclose(statistics.fmean(cpu),summary['other_cpu_cores_mean'],rel_tol=1e-12) and statistics.fmean(cpu)<=4
    return dict(frames=len(frames),sampled_own_peak_mib=peak,other_cpu_cores_mean=statistics.fmean(cpu),sm_clock_mhz_range=[min(clocks),max(clocks)] if clocks else None,temperature_c_range=[min(temperatures),max(temperatures)] if temperatures else None,qualification='Sampled whole-process conditions, not a proof of exclusive access or exact physical peak; CPU eligibility uses mean global other-core activity, not dedicated-core reservation.')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--ledger-root',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();manifest=read(a.run/'manifest.json');assert manifest['status']=='calibration-complete-analysis-pending'
    assert digest(a.queries)=='150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091'
    assert digest(a.oracle)=='7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed'
    qids=[line.split()[0] for line in a.queries.read_text().splitlines()];assert len(qids)==len(set(qids))==500
    gold={q:[] for q in qids}
    for row in csv.DictReader(a.oracle.open(),delimiter='\t'):gold[row['query_id']].append((int(row['original_id']),int(row['score'])))
    assert all(len(v)==10 for v in gold.values());matrices={s:{} for s in SYSTEMS};checks=[];correctness=[];rejections=0;ranks=0
    assert read(a.run/'logs/full-id-sequence.stdout')==dict(documents=8841823,ordered_original_ids_equal=True,gsp_vocab=28679,gsp_nnz=1049282186)
    for entry in manifest['completed']:
        label=entry['label'];timing=entry['tag'].startswith('timing-');block=entry.get('process_block');assert label in matrices
        e=event(a.ledger_root,entry['run'])['payload']
        if entry['tag'].startswith('reject-'):
            assert e['status']=='SPEC_VIOLATION' and set(e['runtime_delta'])=={'config.seed_ranges'} and e['runtime']['config']['seed_ranges']==64
            directory=a.run/'raw'/entry['tag'];assert not (directory/'timing.csv').exists() and not (directory/'top.tsv').exists();rejections+=1;continue
        assert e['status']=='EXECUTED' and e['exit_code']==0
        plan=event(a.ledger_root,e['experiment'])['payload'];assert e['runtime']==dict(config=plan['expected'],behavior=plan['behavior']) and not e['runtime_delta']
        cfg=e['runtime']['config'];assert cfg['docs']==8841823 and cfg['query_count']==500 and cfg.get('k',cfg.get('requested_k'))==10
        inputs={x['role']:x['sha256'] for x in e['inputs']};assert inputs['canonical_gsp']=='e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac' and inputs['queries']==digest(a.queries) and inputs['oracle']==digest(a.oracle)
        assert 'GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7' in e['environment']['gpu']['stdout']
        directory=a.run/'raw'/entry['tag'];monitor=replay_monitor(directory,label) if timing else None
        csvpath=directory/('samples.tsv' if label=='bmp' else 'timing.csv');tops=directory/('results.tsv' if label=='bmp' else 'top.tsv');times={q:[] for q in qids};sequence=[]
        for row in csv.DictReader(csvpath.open(),delimiter='\t' if label=='bmp' else ','):
            sequence.append((row['phase'],int(row['round']),row['query_id']));wall=int(row['elapsed_ns'])/1e6 if label=='bmp' else float(row['query_wall_ms']);assert math.isfinite(wall) and wall>0
            if row['phase']=='measured':times[row['query_id']].append(wall)
        wanted=[(phase,r,q) for phase,count in [('warmup',3 if timing else 0),('measured',20 if timing else 1)] for r in range(count) for q in qids];assert sequence==wanted
        groups={}
        for row in csv.DictReader(tops.open(),delimiter='\t'):
            key=(row['phase'],int(row['round']),row['query_id']);values=groups.setdefault(key,[]);assert int(row['rank'])==len(values)+1
            score=decimal.Decimal(row['score']);assert score.is_finite() and score==score.to_integral_value() and 0<=score<2**24
            values.append((int(row['document_id'] if label=='bmp' else row['original_id']),int(score)))
        assert list(groups)==wanted and all(v==gold[key[2]] for key,v in groups.items());ranks+=sum(map(len,groups.values()))
        if timing and label=='bmp':
            rows=list(csv.DictReader((directory/'monitoring.tsv').open(),delimiter='\t'));assert len(rows)==46;total=wait=0
            for before,after in zip(rows[::2],rows[1::2]):
                if before['phase']!='measured':continue
                delta=[int(y)-int(x) for x,y in zip(before['cpu_jiffies'].split()[1:9],after['cpu_jiffies'].split()[1:9])];assert len(delta)==8 and min(delta)>=0;total+=sum(delta);wait+=delta[4]
            assert total>0 and wait/total<=.01;monitor['measured_iowait_share']=wait/total
        assert all(len(v)==(20 if timing else 1) for v in times.values())
        if timing:
            assert block not in matrices[label];matrices[label][block]=[statistics.median(times[q]) for q in qids];checks.append(dict(label=label,block=block,run=entry['run'],monitor=monitor,query_groups=len(groups)))
        else:correctness.append(dict(label=label,run=entry['run'],query_groups=len(groups)))
    assert rejections==1 and len(correctness)==5 and {c['label'] for c in correctness}==set(SYSTEMS)
    assert len(checks)==15 and all(set(v)=={0,1,2} for v in matrices.values())
    summaries={}
    for label,blocks in matrices.items():
        values=[x for block in range(3) for x in blocks[block]];summaries[label]=dict(median_ms=statistics.median(values),mean_ms=statistics.fmean(values),p95_ms=quantile(values,.95),p99_ms=quantile(values,.99),max_ms=max(values),per_process_median_ms=[statistics.median(blocks[i]) for i in range(3)])
    contrasts={}
    for left,right,kind in CONTRASTS:contrasts[left+' / '+right]=dict(contrast([matrices[left][i] for i in range(3)],[matrices[right][i] for i in range(3)]),kind=kind)
    result=dict(source_commit=manifest['source_commit'],queries=500,documents=8841823,k=10,batch_size=1,processes=3,warmups=3,rounds=20,ranked_scores_checked=ranks,validation=checks,correctness=correctness,runtime_rejections=rejections,descriptive=summaries,contrasts=contrasts,limitations=['Conditional on three process blocks and one shared-host study; process pairs may be separated by resource waits','Six individual intervals, two primary and four secondary; no family-wise error control claim','Sampled context absence is not exclusivity; global mean CPU rule does not guarantee CPU0 exclusivity','Clock and temperature ranges describe whole processes, not only measured query regions','No heldout or unrestricted floating-weight claim; claim-time comparisons and physical JIT audit must accompany final conclusions'])
    with a.output.open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='RAW_REPLAY_PASS_COMPARATIVE_REVIEW_PENDING',ranked_scores=ranks,contrasts_computed=len(contrasts))))
if __name__=='__main__':main()
