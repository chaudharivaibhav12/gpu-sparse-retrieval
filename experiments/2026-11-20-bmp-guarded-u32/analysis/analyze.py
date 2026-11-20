"""Independent raw replay and fixed three-contrast analysis. No controller latencies."""
import argparse,csv,decimal,hashlib,importlib.util,json,math,re,statistics,sys
from pathlib import Path
SYSTEMS=['bmp64','bmp32','prunets256']
CONTRASTS=[('bmp32','bmp64','primary'),('prunets256','bmp32','secondary'),('prunets256','bmp64','secondary')]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--ledger-root',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    parent=load('frozen_analysis',a.repository/'experiments/2026-11-18-matched-final-evaluation/analysis/analyze_calibration.py');reader=load('query_reader',a.repository/'experiments/2026-11-15-range-worker-sweep/code/expected.py');cpu=load('cpu_expected',Path(__file__).resolve().parents[1]/'code/expected_cpu.py')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=read(a.run/'manifest.json');assert m['status']=='calibration-complete-independent-analysis-pending' and len(m['completed'])==14
    assert sha(a.queries)==m['inputs']['queries']['sha256']=='150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091';assert sha(a.oracle)==m['inputs']['oracle']['sha256']=='7a63bb5214891b0c820d7d8205a16701b473292f2f3bf6b05fc0c2494ff2b6ed'
    rows=reader.queries(a.queries,28679,501,1);qids=[q for q,_ in rows];assert len(qids)==len(set(qids))==500;maxterms=max(len(t) for _,t in rows);maxbound=max(255*sum(w for _,w in t) for _,t in rows);gold={q:[] for q in qids}
    with a.oracle.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):gold[row['query_id']].append((int(row['original_id']),int(row['score'])))
    assert all(len(v)==10 for v in gold.values());assert sha(a.run/'calibration-500.pisa')=='0ac4ad355c198fffa65a94ec4ae23da19a0fbea5978a690f343e850fdb7a00d9';matrices={s:{} for s in SYSTEMS};checks=[];negative=[];correct=[];ranks=0;artifact_count=0
    for entry in m['completed']:
        label=entry['label'];is_cpu=label.startswith('bmp');bits=int(label[3:]) if is_cpu else None;directory=a.run/'raw'/entry['tag'];assert directory.is_dir();e=parent.event(a.ledger_root,entry['run'])['payload'];plan=parent.event(a.ledger_root,e['experiment'])['payload']
        for item in e['artifacts']:assert sha(a.ledger_root/item['blob'])==item['sha256'];artifact_count+=1
        runtime=e['runtime'];assert runtime is not None;cfg=runtime['config'];assert cfg['docs']==8841823 and cfg['query_count']==500 and cfg.get('k',cfg.get('requested_k'))==10
        if not is_cpu:
            frozen=read(a.repository/'experiments/2026-11-18-matched-final-evaluation/configs/pruned-runtime-templates.json')['templates']['256']['runtime'];variable={'queries_path','query_count','max_query_terms','max_conservative_score_bound','repeats','warmup_rounds','verify_scores'}
            assert set(cfg)==set(frozen['config']) and runtime['behavior']==frozen['behavior'] and all(cfg[k]==v for k,v in frozen['config'].items() if k not in variable)
            assert cfg['max_query_terms']==maxterms and cfg['max_conservative_score_bound']==maxbound
        if is_cpu:
            state=cpu.expected(m['inputs']['bmp_index']['path'],cfg['queries_path'],8841823,28679,1049282186,500,maxterms,maxbound,bits,cfg['repeats'],cfg['warmups']);assert runtime==state
        roles={x['role']:x['sha256'] for x in e['inputs']};assert roles['queries']==sha(a.queries) and roles['oracle']==sha(a.oracle) and roles['canonical_gsp']=='e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac'
        assert 'GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7' in e['environment']['gpu']['stdout']
        if is_cpu:
            assert roles['prepared_queries']==sha(a.run/'calibration-500.pisa')
            audit=read(directory/'validation.json');assert audit==dict(docs=8841823,vocab=28679,nnz=1049282186,block_size=32,score_bits=bits,full_forward_unique=True,raw_bounds_conservative=True,query_count=500,max_query_terms=maxterms,max_conservative_score_bound=maxbound)
        timing=entry['tag'].startswith('timing-');reject=entry['tag'].startswith('reject-')
        if reject:
            assert is_cpu and e['status']=='SPEC_VIOLATION' and set(e['runtime_delta'])=={'config.score_bits'}
            assert all(len((directory/n).read_text().splitlines())==1 for n in ['samples.tsv','results.tsv','monitoring.tsv']) and not (directory/'memory.json').exists();negative.append(label);continue
        assert e['status']=='EXECUTED' and e['exit_code']==0 and not e['runtime_delta'] and runtime==dict(config=plan['expected'],behavior=plan['behavior'])
        csvfile=directory/('samples.tsv' if is_cpu else 'timing.csv');top=directory/('results.tsv' if is_cpu else 'top.tsv');times={q:[] for q in qids};wanted=[(phase,str(r),q) for phase,n in [('warmup',3 if timing else 0),('measured',20 if timing else 1)] for r in range(n) for q in qids];sequence=[]
        with csvfile.open() as f:
            for row in csv.DictReader(f,delimiter='\t' if is_cpu else ','):
                sequence.append((row['phase'],row['round'],row['query_id']));v=int(row['elapsed_ns'])/1e6 if is_cpu else float(row['query_wall_ms']);assert math.isfinite(v) and v>0
                if row['phase']=='measured':times[row['query_id']].append(v)
        assert sequence==wanted;groups={}
        with top.open() as f:
            for row in csv.DictReader(f,delimiter='\t'):
                key=(row['phase'],row['round'],row['query_id']);items=groups.setdefault(key,[]);assert int(row['rank'])==len(items)+1;score=decimal.Decimal(row['score']);assert score.is_finite() and score==score.to_integral_value() and 0<=score<=1044480;items.append((int(row['document_id'] if is_cpu else row['original_id']),int(score)))
        assert list(groups)==wanted and all(v==gold[k[2]] for k,v in groups.items());ranks+=sum(map(len,groups.values()))
        monitor=parent.replay_monitor(directory,'bmp' if is_cpu else label) if timing else None
        if is_cpu:
            audit=read(directory/'validation.json');assert audit==dict(docs=8841823,vocab=28679,nnz=1049282186,block_size=32,score_bits=bits,full_forward_unique=True,raw_bounds_conservative=True,query_count=500,max_query_terms=maxterms,max_conservative_score_bound=maxbound)
            memory=read(directory/'memory.json');assert memory['score_bytes']==bits//8 and memory['process_vm_swap_kib']<=1024**2
            if timing:
                with (directory/'monitoring.tsv').open() as f:frames=list(csv.DictReader(f,delimiter='\t'))
                assert len(frames)==46;total=wait=0
                for before,after in zip(frames[::2],frames[1::2]):
                    assert before['position']=='before' and after['position']=='after' and (before['phase'],before['round'])==(after['phase'],after['round'])
                    if before['phase']!='measured':continue
                    delta=[int(y)-int(x) for x,y in zip(before['cpu_jiffies'].split()[1:9],after['cpu_jiffies'].split()[1:9])];assert len(delta)==8 and min(delta)>=0;total+=sum(delta);wait+=delta[4]
                assert total>0 and wait/total<=.01;monitor['measured_iowait_share']=wait/total;monitor['memory']=memory
        if timing:
            block=entry['process_block'];assert re.fullmatch(r'timing-p'+str(block)+'-'+label+r'(?:-resource-retry[12])?',entry['tag']);assert block not in matrices[label];assert all(len(v)==20 for v in times.values());matrices[label][block]=[statistics.median(times[q]) for q in qids];checks.append(dict(label=label,block=block,run=entry['run'],monitor=monitor))
        else:correct.append(label)
    assert negative==SYSTEMS[:2] and correct==SYSTEMS and len(checks)==9 and ranks==1050000 and all(set(v)=={0,1,2} for v in matrices.values())
    assert [(x['block'],x['label']) for x in checks]==[(b,s) for b in range(3) for s in SYSTEMS[b:]+SYSTEMS[:b]]
    descriptive={}
    for s,blocks in matrices.items():
        values=[v for b in range(3) for v in blocks[b]];descriptive[s]=dict(median_ms=statistics.median(values),mean_ms=statistics.fmean(values),p95_ms=parent.quantile(values,.95),p99_ms=parent.quantile(values,.99),max_ms=max(values),process_medians_ms=[statistics.median(blocks[b]) for b in range(3)])
    contrasts={left+' / '+right:dict(parent.contrast([matrices[left][i] for i in range(3)],[matrices[right][i] for i in range(3)]),kind=kind) for left,right,kind in CONTRASTS}
    result=dict(status='RAW_REPLAY_PASS_CLAIM_TIME_REVIEW_PENDING',queries=500,documents=8841823,k=10,batch_size=1,ranked_scores_checked=ranks,runner_artifact_hashes=artifact_count,validation=checks,descriptive=descriptive,contrasts=contrasts,source_commit=m['source_commit'],cpu_backend_source_commit=m['cpu_backend_source_commit'],gpu_backend_source_commit=m['gpu_backend_source_commit'],limitations=['Calibration-only;6480parentqueries exposed but notused fordevelopment','Individual intervals conditional on three assignedsharedhost processblocks;no familywise guarantee','Claim-time comparisons/build-source review stillrequired,including arithmetic labels','Freshstrongeractualindexpreflight is outside query timing forbothCPUvariants','U32neednotimprove latency or totalprocessmemory;candidate/heap padding andindexmemory disclosed','No strongestCPU/releasedBMP/generalGPUclaim'])
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'ranked_scores':ranks,'contrasts':len(contrasts)}))
if __name__=='__main__':main()
