"""One fixed five-system full-collection calibration study through the team runner.

This entry point evaluates heldout queries only after an immutable method freeze
and instrumented independent oracle generation. No query-dependent tuning. Resource waits are outside all query timing.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,math,os,signal,statistics,subprocess,sys,threading,time,traceback
from pathlib import Path
from pruned_expected import expected as pruned_expected
E='2026-11-18-matched-final-evaluation'
TARGET='GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7'
BOUNDARY='Resident parsed cap32 term/weight pairs through fresh query preparation, complete exact retrieval/selection/transfers/synchronization, and ordered original numeric IDs and integer scores in host memory; excludes file decoding, index setup, domain validation and textual formatting.'
ORDER=['prunets256','prunets1024','range','compact','bmp']
class ResourceIneligible(RuntimeError):pass
def admission_command(root,log,argv):
    return [sys.executable,Path(root)/'experiments'/E/'code/admit_exec.py','--log',log,*argv]

def resource_retries(run,label,tag,*args):
    for retry in range(3):
        try:return run(label,tag+('' if retry==0 else '-resource-retry'+str(retry)),*args)
        except ResourceIneligible:
            if retry==2:raise
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gpu_state():
    p=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free,clocks.sm,pstate,power.draw,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    rows=list(csv.reader(p.stdout.splitlines()));assert len(rows)==1 and len(rows[0])==7 and rows[0][0].strip()==TARGET
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    return dict(telemetry=dict(zip(['sm_clock_mhz','pstate','power_w','utilization_percent','temperature_c'],[v.strip() for v in rows[0][2:]])),free_mib=int(rows[0][1]),contexts=[(int(r[1]),int(r[2])) for r in csv.reader(p.stdout.splitlines()) if r and r[0].strip()==TARGET])
def available_kib():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
def busy_ticks():
    v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));return sum(v)-v[3]-v[4]
def cpu0_ticks():
    values=list(map(int,next(x for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu0 ')).split()[1:9]));assert len(values)==8
    return sum(values)-values[3]-values[4]
def own_ticks(group,seen):
    swap=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            v=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(v[2])!=group:continue
            key=(int(p.name),int(v[19]));seen[key]=max(seen.get(key,0),int(v[11])+int(v[12]))
            status=(p/'status').read_text();swap+=int(next((x.split()[1] for x in status.splitlines() if x.startswith('VmSwap:')),'0'))
        except (FileNotFoundError,ProcessLookupError):continue
    return sum(seen.values()),swap
def monitor(stop,path,group,result,process):
    seen={};last=None;samples=0;own_context=False;foreign=set();errors=[];cpu=[];peak=0;swapped=False;ticks=os.sysconf('SC_CLK_TCK')
    with path.open('x') as output:
        while not stop.is_set():
            frame={'utc':utc(),'run_process_group':group,'ticks_per_second':ticks}
            try:
                state=gpu_state();frame['gpu']=state;frame['context_ownership']=[];memory=0
                for pid,mib in state['contexts']:
                    try:pgid=os.getpgid(pid);owned=pgid==group
                    except ProcessLookupError:pgid=None;owned=False
                    frame['context_ownership'].append(dict(pid=pid,pgid=pgid,owned=owned,used_mib=mib,classification='resolved-process-group' if pgid is not None else 'unclassified-exited'))
                    if owned:memory+=mib;own_context=True
                    else:foreign.add(pid)
                peak=max(peak,memory);own,swap=own_ticks(group,seen);frame['own_swap_kib']=swap
                if swap>1024**2:
                    swapped=True
                    # The process group was created by this controller for this run.
                    # Never act on any group discovered through nvidia-smi.
                    if process.poll() is None:
                        try:os.killpg(group,signal.SIGTERM)
                        except ProcessLookupError:pass
                current=(time.monotonic(),busy_ticks(),own);frame.update(sample_monotonic=current[0],machine_busy_ticks=current[1],own_group_cpu_ticks=own,cpu0_busy_ticks=cpu0_ticks())
                try:frame['cpu0_frequency_khz']=Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq').read_text().strip()
                except OSError:frame['cpu0_frequency_khz']=None
                if last:
                    dt=current[0]-last[0];other=max(0.0,((current[1]-last[1])-(current[2]-last[2]))/ticks/dt);cpu.append(other);frame['other_cpu_cores']=other
                last=current;samples+=1
            except Exception as error:errors.append(repr(error));frame['error']=repr(error)
            output.write(json.dumps(frame)+'\n');output.flush();stop.wait(1)
    result.update(run_process_group=group,ticks_per_second=ticks,samples=samples,own_gpu_context_seen=own_context,foreign_pids=sorted(foreign),errors=errors,other_cpu_cores_mean=statistics.fmean(cpu) if cpu else None,sampled_own_peak_mib=peak,own_swap_limit_exceeded=swapped,qualification='One-second observations are sampled evidence, not proof of exclusive access or exact memory peaks. CPU estimate conservatively includes controller and monitoring costs.')
def validate_outputs(label,paths,gold,qids,repeats,warmups,verify=0,config=None):
    delimiter='\t' if label=='bmp' else ','
    with paths['csv'].open(newline='') as f:rows=list(csv.DictReader(f,delimiter=delimiter))
    expected=[(phase,str(r),q) for phase,n in [('warmup',warmups),('measured',repeats)] for r in range(n) for q in qids]
    assert [(r['phase'],r['round'],r['query_id']) for r in rows]==expected
    lat={q:[] for q in qids}
    for row in rows:
        value=int(row['elapsed_ns'])/1e6 if label=='bmp' else float(row['query_wall_ms'])
        assert math.isfinite(value) and value>0
        if row['phase']=='measured':lat[row['query_id']].append(value)
        if label!='bmp':assert int(row['docs'])==8841823 and int(row['k'])==10
        for key in ['ordered_id_mismatches','result_score_mismatches','schedule_mismatches']:
            if key in row:assert row[key]==('0' if verify else 'not_checked')
        if label=='compact':
            assert row['variant']=='monolithic' and int(row['result_count'])==10 and int(row['position_count'])==10
            for key in ['invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors','canonical_id_mismatches']:assert int(row[key])==0
            assert row['true_score_errors']==('0' if verify else 'post_validation')
        if config:
            for key in ['range_size','score_threads','merge_threads','index_posting_bytes','device_allocation_bytes']:
                if key in config and key in row:assert int(row[key])==config[key]
    groups={}
    with paths['top'].open(newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            key=(row['phase'],row['round'],row['query_id']);v=groups.setdefault(key,[]);assert int(row['rank'])==len(v)+1
            score=float(row['score']);assert math.isfinite(score) and score.is_integer() and 0<=score<2**24
            v.append((int(row['document_id'] if label=='bmp' else row['original_id']),int(score)))
    assert list(groups)==expected
    for key,v in groups.items():assert v==gold[key[2]],key
    io_share=None
    if label=='bmp':
        with paths['cpu_monitor'].open(newline='') as f:monitoring=list(csv.DictReader(f,delimiter='\t'))
        assert len(monitoring)==2*(warmups+repeats);total=wait=0
        for before,after in zip(monitoring[::2],monitoring[1::2]):
            assert before['position']=='before' and after['position']=='after' and (before['phase'],before['round'])==(after['phase'],after['round'])
            if before['phase']!='measured':continue
            delta=[int(b)-int(a) for a,b in zip(before['cpu_jiffies'].split()[1:9],after['cpu_jiffies'].split()[1:9])];assert len(delta)==8 and min(delta)>=0;total+=sum(delta);wait+=delta[4]
        assert total>0;io_share=wait/total
    return {q:statistics.median(v) for q,v in lat.items()},dict(groups=len(groups),ranked_scores=sum(map(len,groups.values())),iowait_share=io_share)
def validate_method_state(cfg,behavior,template):
    assert behavior==template['behavior'] and set(cfg)==set(template['config'])
    variable={'queries_path','query_count','max_query_terms','max_conservative_score_bound','query_tensor_width','reference_path'}
    assert all(cfg[k]==v for k,v in template['config'].items() if k not in variable),'Frozen algorithm setting drift'
    assert cfg['query_count']==6480 and cfg.get('max_query_terms',0)<=128 and cfg.get('max_conservative_score_bound',0)<2**24

def main():
    def terminate_owned(signum,frame):raise KeyboardInterrupt('Owned controller termination')
    signal.signal(signal.SIGTERM,terminate_owned)
    parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    root=a.repository.resolve();out=a.output.resolve();assert out.is_relative_to(root);os.chdir(root);sys.path.insert(0,str(root/'scripts'))
    from research import Ledger,digest,read,write_new,review,export,differences
    inputs=read(a.inputs);receipt=read(root/'deployment-receipt.json');assert receipt['source_commit']==inputs['source_commit']
    out.mkdir(parents=True,exist_ok=False)
    for name in ['logs','raw','analysis','build','jit','cuda-cache']:(out/name).mkdir()
    write_new(out/'heldout-inputs.json',inputs)
    L=Ledger(root);completed=[];attempts=[];deadline=time.monotonic()+259200
    def status(state,**extra):
        # Status is mutable operational state; scientific records remain append-only.
        temp=out/'status.pending';temp.write_text(json.dumps(dict(state=state,utc=utc(),source_commit=inputs['source_commit'],completed=len(completed),**extra),indent=2));temp.replace(out/'status.json')
    def source_check():
        for name,proof in {**receipt['files'],**receipt.get('support_files',{})}.items():assert digest(root/name)==(proof['sha256'] if isinstance(proof,dict) else proof),name
        manifest=inputs['build_manifest'];assert digest(manifest['path'])==manifest['sha256']
        for binary in read(manifest['path'])['binaries'].values():assert digest(binary['path'])==binary['sha256']
    def wait_resources(tag):
        while True:
            source_check();start=busy_ticks();t=time.monotonic();time.sleep(2);other=(busy_ticks()-start)/os.sysconf('SC_CLK_TCK')/(time.monotonic()-t);state=gpu_state();memory=available_kib()
            record=dict(utc=utc(),tag=tag,gpu=state,host_available_kib=memory,other_cpu_cores=other)
            with (out/'logs/availability.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if not state['contexts'] and state['free_mib']>=10000 and memory>=34*1024**2 and other<=4:return
            if time.monotonic()>=deadline:raise RuntimeError('Resource admission deadline; no foreign job affected')
            status('waiting-for-resources',next_stage=tag,observation=record);time.sleep(120)
    def observed(argv,tag,sensitive=False):
        stop=threading.Event();summary={};observer=None
        with (out/'logs'/(tag+'.stdout')).open('xb') as so,(out/'logs'/(tag+'.stderr')).open('xb') as se:
            p=subprocess.Popen(list(map(str,argv)),cwd=root,stdout=so,stderr=se,env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1'),start_new_session=True)
            if sensitive:
                observer=threading.Thread(target=monitor,args=(stop,out/'logs'/(tag+'.monitor.jsonl'),p.pid,summary,p),daemon=True);observer.start()
            try:p.wait(timeout=1800)
            except BaseException:
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
            finally:
                stop.set()
                if observer:observer.join(timeout=45)
        e=L.event('observation',dict(title='Matched study preparation '+tag,motivation='Bind uninstrumented setup/build to exact source and output before retrieval',argv=list(map(str,argv)),cwd=str(root),exit_code=p.returncode,monitor=summary,qualification='Uninstrumented setup/data validation, no retrieval performance conclusion',artifacts=[L.blob(out/'logs'/(tag+'.stdout')),L.blob(out/'logs'/(tag+'.stderr'))]),[question['id']]);assert p.returncode==0 and (not observer or not observer.is_alive()),(tag,e['id'])
    try:
        source_check()
        question=L.event('question',dict(title='Frozen full-collection heldout comparison',motivation='Close cross-session and CPU-formatting boundary gaps using strongest audited exact controls before heldout freeze',source_commit=inputs['source_commit'],parent_merge='b31ff7a57b3283884d637a8f8480e85640ae9242',protocol=read(root/'experiments'/E/'configs/heldout-method.json'),artifacts=[L.blob(a.inputs),L.blob(root/'deployment-receipt.json')]),[])
        wait_resources('before-large-input-hash-and-setup');status('validating-full-input-hashes')
        # Calibration paths/hashes are fixed explicitly. Never substitute another query file.
        assert inputs['queries']['sha256']==inputs['prepared_queries_sha256']
        assert read(root/'research/events'/ (inputs['method_freeze']+'.json'))['kind']=='freeze'
        assert digest(inputs['query_split']['path'])==inputs['query_split']['sha256']
        for item in [inputs['gsp'],inputs['queries'],inputs['oracle'],inputs['bmp_index'],inputs['rbp256'],inputs['rbp1024']]:assert digest(item['path'])==item['sha256'],item['path']
        assert inputs['gsp']['sha256']=='e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac'
        assert inputs['oracle']['sha256']==inputs['prepared_oracle_sha256']
        assert inputs['bmp_index']['sha256']=='eaf549b82636dba01287ebd7e9128244f1cbc8d85af233c9587a4a7c205df94d'
        assert digest(Path(inputs['native_repo'])/'src/triton_kernel.py')==inputs['native_monolithic_sha256']
        assert subprocess.check_output(['git','-C',inputs['native_repo'],'rev-parse','HEAD'],text=True).strip()==inputs['native_revision']=='92bb87b36f10e78ab39c81e9db2412dbb7a3b650'
        re=module('range_expected',root/'experiments/2026-11-15-range-worker-sweep/code/expected.py')
        sys.path.insert(0,str(root/'experiments/2026-11-17-gpusparse-u8-storage/code'))
        fe=module('compact_expected',root/'experiments/2026-11-17-gpusparse-u8-storage/code/fp32/expected_config.py')
        exporter=module('bmp_export',root/'experiments/2026-11-18-bmp-full-collection/code/export_queries.py')
        gsp=inputs['gsp'];gm=dict(docs=8841823,vocab=28679,nnz=1049282186,multiplicity=1,sha256=gsp['sha256'],path=gsp['path'])
        # Reuse previously reviewed whole-index multiplicity/encoding evidence by hash.
        domain=inputs['domain_validation'];assert digest(domain['path'])==domain['sha256'];certificate=read(domain['path'])['payload']
        assert certificate['canonical_gsp_sha256']==gm['sha256'] and certificate['maximum_term_document_multiplicity']==1 and certificate['queries_sha256']==inputs['calibration_queries_sha256']
        L.event('observation',dict(title='Reuse pinned full-index domain validation',motivation='Same immutable full GSP and calibration queries; unique document terms and exact uint8 representation were independently reviewed in the parent study. Fresh outputs still required.',original_record=read(domain['path'])['id'],qualification='Reused historical validation, not a fresh scan of every posting',artifacts=[L.blob(domain['path'])]),[question['id']])
        query=Path(inputs['queries']['path']);rows=re.queries(query,28679,6481,1);assert len(rows)==6480;qids=[q for q,_ in rows]
        gold={}
        with open(inputs['oracle']['path']) as f:
            for r in csv.DictReader(f,delimiter='\t'):gold.setdefault(r['query_id'],[]).append((int(r['original_id']),int(r['score'])))
        assert list(gold)==qids and all(len(v)==10 for v in gold.values())
        _,vocab=exporter.vocabulary(Path(gsp['path']));assert len(set(vocab))==len(vocab)==28679
        pisa=out/'heldout-6480.pisa'
        with pisa.open('x') as f:
            for q,terms in rows:
                assert all(1<=w<=32 for _,w in terms);f.write(q+': '+' '.join(vocab[t] for t,w in terms for _ in range(w))+'\n')
        assert digest(pisa)==inputs['prepared_pisa_sha256']
        metadata={w:re.rbp_metadata(inputs['rbp'+str(w)]['path']) for w in [256,1024]}
        status('verify-prepared-builds')
        assert digest(inputs['build_manifest']['path'])==inputs['build_manifest']['sha256'];prepared=read(inputs['build_manifest']['path']);assert prepared['source_commit']==inputs['backend_source_commit'] and prepared['status']=='complete'
        for binary in prepared['binaries'].values():assert digest(binary['path'])==binary['sha256']
        for log in prepared['logs']:
            assert digest(log['path'])==log['sha256']
        L.event('observation',dict(title='Reviewed prebuilt exact-source binaries and compiler evidence',motivation='Reuse CPU-only compilation at this exact source commit; keep build and measurement identities distinct',build=prepared,artifacts=[L.blob(inputs['build_manifest']['path'])]+[L.blob(x['path']) for x in prepared['logs']]),[question['id']])
        inputs['bmp_binary']=dict(prepared['binaries']['bmp'],adapter_sha256=prepared['adapter_sha256'])
        wait_resources('full-original-passage-ID-sequence-check');status('validating-full-original-ID-sequence')
        observed(['numactl','--physcpubind=0','--localalloc',prepared['binaries']['verify_ids']['path'],inputs['bmp_index']['path'],gsp['path']],'full-id-sequence',True)
        ids=read(out/'logs/full-id-sequence.stdout');assert ids==dict(documents=8841823,ordered_original_ids_equal=True,gsp_vocab=28679,gsp_nnz=1049282186)
        adapter=root/'experiments/2026-11-17-gpusparse-u8-storage/code/fp32/canonical_baseline.py'
        probe="import json,numpy,scipy,torch,triton,hashlib;from pathlib import Path;p=Path(torch.__file__).parent;files=[Path(torch.__file__),*p.glob('_C*.so'),*[p/'lib'/n for n in ['libtorch_cuda.so','libtorch_cpu.so','libc10.so','libc10_cuda.so']]];hashes={str(f):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in files if f.is_file()};print(json.dumps(dict(torch_version=str(torch.__version__),triton_version=triton.__version__,numpy_version=numpy.__version__,scipy_version=scipy.__version__,torch_cuda_version=torch.version.cuda,library_sha256=hashes)))"
        observed([sys.executable,'-c',probe],'packages-before');environment=read(out/'logs/packages-before.stdout');assert environment==inputs['packages']
        device_profile=read(root/'experiments'/E/'configs/device-profile.json');assert device_profile['target_uuid']==TARGET
        resolver_environment=dict(environment,gpu_name=device_profile['gpu_name'],compute_capability=device_profile['compute_capability'])
        assert digest(inputs['bmp_binary']['path'])==inputs['bmp_binary']['sha256']
        assert digest(root/'experiments'/E/'code/project_benchmark_matched.rs')==inputs['bmp_binary']['adapter_sha256']
        builds={name:dict(source_commit=inputs['backend_source_commit'],controller_source_commit=inputs['source_commit'],prepared_build_manifest_sha256=inputs['build_manifest']['sha256'],binary_sha256=prepared['binaries']['bmp' if name=='bmp' else 'prunets' if name.startswith('prunets') else 'range']['sha256'] if name!='compact' else None,packages=environment if name=='compact' else None) for name in ORDER}
        def one(label,tag,repeats,warmups,verify,negative=False,plan_only=False):
            if not plan_only:wait_resources(tag);status('running',stage=tag)
            else:source_check()
            directory=out/'raw'/tag;directory.mkdir()
            paths=dict(csv=directory/('samples.tsv' if label=='bmp' else 'timing.csv'),top=directory/('results.tsv' if label=='bmp' else 'top.tsv'),cpu_monitor=directory/'monitoring.tsv')
            if label.startswith('prunets'):
                w=int(label[len('prunets'):]);index=inputs['rbp'+str(w)]['path'];state=pruned_expected(root,index,query,metadata[w],rows,repeats,warmups,verify);cfg=state['config'];behavior=state['behavior']
                argv=[prepared['binaries']['prunets']['path'],index,query,paths['csv'],paths['top'],6480,10,0,verify,repeats,warmups,256,256,gsp['path'] if verify else '-',64]
            elif label=='range':
                index=inputs['rbp1024']['path'];cfg=re.config(index,query,metadata[1024],gm,rows,10,256,repeats,warmups,verify,0);behavior=re.marks();argv=[prepared['binaries']['range']['path'],index,query,paths['csv'],paths['top'],6480,10,0,verify,repeats,warmups,256,256,gsp['path'] if verify else '-']
            elif label=='compact':
                cfg=fe.expected(gsp['path'],query,inputs['oracle']['path'],inputs['native_repo'],adapter,resolver_environment,0,6480,10,512,repeats,warmups,verify,{'triton_kernel.py':inputs['native_monolithic_sha256']},inputs['native_revision']);behavior=fe.marks(verify)
                argv=[sys.executable,adapter,'--variant','monolithic','--index',gsp['path'],'--queries',query,'--reference',inputs['oracle']['path'],'--native-repo',inputs['native_repo'],'--csv',paths['csv'],'--top',paths['top'],'--validation',directory/'native-validation.json','--jit-manifest',directory/'jit-manifest.json','--limit',6480,'--k',10,'--gpu',0,'--block-pl',512,'--repeats',repeats,'--warmups',warmups,'--verify',verify]
            else:
                cfg=dict(mode='bmp-resident-parsed',docs=8841823,query_count=6480,k=10,block_size=32,cpu=0,warmups=warmups,repeats=repeats,rayon_threads=1,index=inputs['bmp_index']['path'],queries_path=str(pisa),alpha=1.0,beta=1.0,order='file-order',boundary='parsed-query-through-host-numeric-topk')
                behavior=dict(contract='GSP-CAP32-U64-NUMERIC-TIES-v1',arithmetic='uint64',document_impacts='uint8-exact',query_policy='BMP-f32-ceil-cap32',tie_order='score_desc_original_numeric_id_asc',eligibility='all_documents',resident_context='numeric-ids-and-zero-order',bounds='raw',selection='sorted-wide-block-candidates')
                argv=[inputs['bmp_binary']['path'],'--index',inputs['bmp_index']['path'],'--queries',pisa,'--expected-docs',8841823,'--expected-queries',6480,'--cpu',0,'--k',10,'--block-size',32,'--warmups',warmups,'--repeats',repeats,'--samples',paths['csv'],'--results',paths['top'],'--monitoring',paths['cpu_monitor']]
            if plan_only:
                path=out/'analysis'/(tag+'.json');write_new(path,dict(expected=cfg,behavior=behavior,command=list(map(str,argv)),qualification='CPU-only expected-state resolution; no retrieval execution'));return path
            if negative:cfg=dict(cfg,seed_ranges=65)
            phase='negative' if negative else 'correctness' if verify else 'timing'
            frozen_state=inputs['evaluation_rule']['runtime_templates'][phase][label]
            validate_method_state(cfg,behavior,frozen_state)
            calibration_parent=inputs['calibration_experiments'][phase][label]
            parent_state=L.get(calibration_parent)['payload']
            deltas=differences(dict(config=parent_state['expected'],behavior=parent_state['behavior']),dict(config=cfg,behavior=behavior))
            freeze=L.event('freeze',dict(title='Exact resolved heldout runtime '+tag,motivation='Mechanical resolution under immutable pre-content method freeze; no algorithm or selection changes',experiment_ref=calibration_parent,declared_changes={k:'Frozen heldout population/path and mechanically derived metadata; no algorithm or semantic delta' for k in deltas},expected=cfg,behavior=behavior,query_split_sha256=inputs['query_split']['sha256'],evaluation_rule=inputs['evaluation_rule'],method_freeze=inputs['method_freeze']),[inputs['method_freeze'],calibration_parent])
            spec=L.plan(dict(title='Frozen heldout '+tag,question=question['id'],hypothesis='Exact same canonical answers; empirical timing differences',motivation='Five fresh controls under one disclosed boundary and admission protocol',owner='vaibhav',expected=cfg,behavior=behavior,split='heldout',freeze_ref=freeze['id'],success_rule='SPEC_VIOLATION before query data' if negative else 'EXECUTED, exact complete outputs, eligible timing where applicable',failure_rule='Preserve and review; up to3 reviewed technical repairs',next_if_null='Retain slower/null result without tuning',timing_boundary=BOUNDARY,inputs=[dict(role=key,path=item['path'],sha256=item['sha256']) for key,item in [('canonical_gsp',gsp),('queries',inputs['queries']),('oracle',inputs['oracle']),('query_split',inputs['query_split'])]]+([dict(role='index',path=inputs['bmp_index']['path'],sha256=inputs['bmp_index']['sha256']),dict(role='prepared_queries',path=str(pisa),sha256=digest(pisa))] if label=='bmp' else [dict(role='index',path=inputs['rbp'+('256' if label=='prunets256' else '1024')]['path'],sha256=inputs['rbp'+('256' if label=='prunets256' else '1024')]['sha256'])] if label!='compact' else []),build=builds[label],correctness_rule='All raw canonical answers match independent full oracle; complete sample/order checks',analysis_plan='Per-query20-round medians; paired process/query geometric ratios; fixed crossed bootstrap; no heldout',refs=[question['id']]))
            command=[sys.executable,root/'scripts/research.py','run',spec['id'],'--gate-seconds',600]
            for name,path in paths.items():
                if name!='cpu_monitor' or label=='bmp':command+=['--output',path]
            if label=='compact':command+=['--output',directory/'native-validation.json','--output',directory/'jit-manifest.json']
            admission=directory/'admission.jsonl'
            command+=['--output',admission,'--command',*admission_command(root,admission,argv)];stop=threading.Event();summary={};observer=None
            wait_resources(tag+'-final-admission');status('starting-instrumented-run',stage=tag,experiment=spec['id'],timing_requested=tag.startswith('timing-'))
            with (directory/'runner.stdout').open('xb') as so,(directory/'runner.stderr').open('xb') as se:
                launch=['numactl','--physcpubind=0','--localalloc',*map(str,command)];write_new(directory/'runner-command.json',dict(argv=launch,cwd=str(root),environment_overrides=dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUDA_MODULE_LOADING='LAZY',TRITON_CACHE_DIR=str(out/'jit'/tag),CUDA_CACHE_PATH=str(out/'cuda-cache'/tag))))
                p=subprocess.Popen(launch,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUDA_MODULE_LOADING='LAZY',TRITON_CACHE_DIR=str(out/'jit'/tag),CUDA_CACHE_PATH=str(out/'cuda-cache'/tag)),stdout=so,stderr=se,start_new_session=True)
                try:
                    write_new(directory/'process-identity.json',dict(runner_pid=p.pid,expected_process_group=p.pid,parent_controller_pid=os.getpid(),created_at_utc=utc(),source_commit=inputs['source_commit'],stage=tag))
                    status('instrumented-run-active',stage=tag,runner_pid=p.pid,experiment=spec['id'],timing_requested=tag.startswith('timing-'))
                    observer=threading.Thread(target=monitor,args=(stop,directory/'monitor.jsonl',p.pid,summary,p),daemon=True);observer.start()
                    rc=p.wait(timeout=86400 if verify else 21600)
                except BaseException:
                    if p.poll() is None:
                        try:os.killpg(p.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        p.wait()
                    raise
                finally:
                    stop.set()
                    if observer is not None and observer.ident is not None:observer.join(timeout=45)
            assert not observer.is_alive();write_new(directory/'monitor-summary.json',summary);event=read(directory/'runner.stdout');done=event['id']
            admission_records=[json.loads(line) for line in admission.read_text().splitlines()] if admission.exists() else []
            admission_timeout=(event['payload'].get('exit_code')==75 and event['payload'].get('runtime') is None and admission_records and admission_records[-1]['decision']=='timeout' and not paths['csv'].exists() and not paths['top'].exists())
            if admission_timeout:
                review(L,done,dict(result_summary=dict(admission_timeout=True,monitor=summary),interpretation='Backend not started: post-hash admission timed out',conclusions=[],decision='Preserve failed runner attempt and replace under resource-only retry cap',motivation='Availability can change during runner hashing; no query compute allowed here',confidence='high',limitations=['Availability is not a reservation'],next_steps=[dict(task='Wait for resources and retry fresh attempt',owner='unassigned',reason='No backend executed',done_when='Post-hash admission succeeds')]))
                entry=dict(tag=tag,label=label,run=done,valid=False,admission_timeout=True,monitor=summary);attempts.append(entry);write_new(out/'analysis'/(tag+'.json'),entry)
                raise ResourceIneligible(tag)
            if negative:
                valid=event['payload']['status']=='SPEC_VIOLATION' and set(event['payload'].get('runtime_delta') or {})=={'config.seed_ranges'} and not paths['csv'].exists() and not paths['top'].exists();lat={};validation={'deliberate_rejection':valid}
            else:
                valid=rc==0 and event['payload']['status']=='EXECUTED';lat={};validation={'status':event['payload']['status']}
                if valid:
                    try:
                        lat,validation=validate_outputs(label,paths,gold,qids,repeats,warmups,verify,cfg)
                        if label=='compact':
                            inspector=module('compact_physical',root/'experiments/2026-11-17-gpusparse-u8-storage/code/inspect_compact_jit.py');write_new(directory/'physical-arithmetic.json',inspector.inspect_compact(out/'jit'/tag))
                    except Exception as error:valid=False;validation={'error':repr(error)}
            timing=tag.startswith('timing-');eligible=not summary['foreign_pids'] and not summary['errors'] and summary['samples']>0 and not summary['own_swap_limit_exceeded']
            if timing:eligible=eligible and summary['other_cpu_cores_mean'] is not None and summary['other_cpu_cores_mean']<=4 and (label=='bmp' or summary['own_gpu_context_seen']) and (label!='bmp' or validation.get('iowait_share',1)<=.01)
            review(L,done,dict(result_summary=dict(validation=validation,monitor=summary,eligible=eligible if timing else None),interpretation='Exactness/runtime checks passed' if valid else 'Failed attempt; preserve and inspect',conclusions=[dict(text='Per-run checks only; no comparative speed claim',scope='Full collection,6480 heldout queries,k10,batch1',evidence_refs=[done])],decision='Continue fixed protocol' if valid and (eligible or not timing) else 'Wait and replace under frozen resource-only retry cap' if valid and timing else 'Stop for reviewed technical repair',motivation='Correctness and uncontaminated timing precede comparison',confidence='high' if valid else 'low',limitations=['Sampled shared-host evidence','No final heldout conclusions'],next_steps=[dict(task='Complete fixed calibration and independent review before freeze',owner='unassigned',reason='Final evaluation prerequisites',done_when='All15 eligible timing processes and canonical outputs reviewed')]))
            entry=dict(tag=tag,label=label,run=done,latencies=lat if timing else None,validation=validation,monitor=summary,valid=valid,timing_eligible=eligible if timing else None);attempts.append(entry);write_new(out/'analysis'/(tag+'.json'),entry)
            assert valid,(tag,validation,summary)
            if timing and not eligible:raise ResourceIneligible(tag)
            completed.append(entry);return entry
        preflight=[]
        for verify,repeats,warmups,phase in [(1,1,0,'correctness'),(0,20,3,'timing')]:
            for label in ORDER:preflight.append(one(label,'config-preflight-'+phase+'-'+label,repeats,warmups,verify,plan_only=True))
        L.event('observation',dict(title='All five expected-state factories resolved before query execution',motivation='Catch incomplete software/hardware configuration assembly before any long correctness stage',artifacts=[L.blob(p) for p in preflight]+[L.blob(root/'experiments'/E/'configs/device-profile.json')]),[question['id']])
        resource_retries(one,'prunets256','reject-prunets-full-state',1,0,0,True)
        for label in ORDER:resource_retries(one,label,'correctness-'+label,1,0,1)
        for block in range(3):
            for label in ORDER[block:]+ORDER[:block]:resource_retries(one,label,f'timing-p{block}-'+label,20,3,0)['process_block']=block
        observed([sys.executable,'-c',probe],'packages-after');assert read(out/'logs/packages-after.stdout')==environment
        write_new(out/'manifest.json',dict(status='heldout-complete-analysis-pending',source_commit=inputs['source_commit'],completed=completed,attempts=attempts,qualification='No comparative claim until independent bootstrap/claim-time comparisons. Compact physical-kernel checks retained for each compact run. Frozen heldout evaluation.'))
        write_new(out/'ledger-verify.json',L.verify());export(L);status('heldout-complete-independent-analysis-pending')
    except BaseException as error:
        events=L.events();reviewed={e['payload'].get('run') for e in events if e['kind']=='review'};pending=[e['id'] for e in events if e['kind']=='run_finished' and e['id'] not in reviewed]
        L.event('observation',dict(title='Frozen heldout stopped',motivation='Preserve failure and unfinished work before reviewed repair',error=repr(error),completed=len(completed),runs_pending_review=pending),pending);status('failed',error=repr(error));traceback.print_exc();raise
if __name__=='__main__':main()
