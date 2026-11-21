"""Fixed three-system calibration study. Explicit source-bound approval required.

Reuses reviewed parent monitor/admission/validators; no heldout access, parameter
search, implicit technical repair or historical timing splice.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,threading,time,traceback
from pathlib import Path
from expected_cpu import expected as cpu_expected
E='2026-11-20-bmp-guarded-u32';P='2026-11-18-matched-final-evaluation'
ORDER=['bmp64','bmp32','prunets256']
BOUNDARY='Resident parsed cap32 term/weight pairs through fresh query preparation, complete exact retrieval/selection/transfers/synchronization, and ordered original numeric IDs and integer scores in host memory; excludes file decoding, index setup, domain validation and textual formatting.'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def schedule():
    return [(b,s) for b in range(3) for s in ORDER[b:]+ORDER[:b]]

def approval(value,source,root,output):
    assert value['approved'] is True and value['experiment']==E and value['source_commit']==source
    assert value['root']==str(root) and value['output']==str(output)
    assert value['scope']=='one-fixed-calibration-study' and value['resource_attempts_per_stage']==3
    assert value['technical_resubmissions']==0 and value['queries']==500 and value['heldout_queries']==0

def validate_protocol(p):
    assert p['systems']==ORDER and p['timing_order']==[ORDER[b:]+ORDER[:b] for b in range(3)]
    assert p['scope']==dict(documents=8841823,queries=500,split='calibration',k=10,batch=1,cpu=0,numa=0,rayon_threads=1,bmp_block_size=32,bounds='raw',alpha=1.0,beta=1.0,gpu_uuid='GPU-ab7b89ba-5664-fd05-e352-a4bc034e5ff7',process_blocks=3,warmups=3,repeats=20)
    assert p['resources']==dict(gpu_free_mib=10000,host_available_gib=34,other_cpu_cores_mean_max=4,bmp_measured_iowait_share_max=.01,own_swap_gib_max=1,resource_attempts_per_stage=3,poll_seconds=120,availability_hours=72,runtime_gate_seconds=600,post_hash_admission_seconds=120,stage_timeout_seconds=21600)
    assert p['environment']==dict(cuda_module='cuda-12.6',python_module='python-3.12',python_interpreter='/scratch/vc2836/gpu-project/sessions/pruning-replication-95ae7f1/tmp/gpusparse-setup-92bb87b-v2/venv/bin/python',python_major_minor=[3,12],OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',CUDA_MODULE_LOADING='LAZY')
    assert p['primary_contrast']==['bmp32','bmp64'] and p['secondary_contrasts']==[['prunets256','bmp32'],['prunets256','bmp64']]

def runner_result(path):
    v=json.loads(Path(path).read_text())
    if not isinstance(v,dict) or not isinstance(v.get('id'),str) or not isinstance(v.get('payload'),dict):raise ValueError('Invalid runner final JSON schema')
    return v

def cpu_audit(path,bits,max_terms,max_bound):
    v=json.loads(Path(path).read_text());assert v==dict(docs=8841823,vocab=28679,nnz=1049282186,block_size=32,score_bits=bits,full_forward_unique=True,raw_bounds_conservative=True,query_count=500,max_query_terms=max_terms,max_conservative_score_bound=max_bound);return v

def group_snapshot(group,directory):
    rows=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=group:continue
            rows.append(dict(pid=int(path.name),uid=path.stat().st_uid,session=int(fields[3]),matches_output=str(directory).encode() in (path/'cmdline').read_bytes()))
        except (FileNotFoundError,ProcessLookupError):continue
    return rows

def signal_own_remainder(group,directory,rows=None,send=None,uid=None):
    # A dead wrapper can leave its backend alive. Never signal a recycled or
    # unresolved group:require ourUID/setsid session and a matching output path.
    try:rows=group_snapshot(group,directory) if rows is None else rows
    except (OSError,ValueError,IndexError) as error:return dict(decision='refused-proc-inspection-error',error=repr(error))
    if not rows:return dict(decision='no-remaining-members')
    uid=os.getuid() if uid is None else uid
    if not all(r['uid']==uid and r['session']==group for r in rows) or not any(r['matches_output'] for r in rows):
        return dict(decision='refused-unresolved-ownership',pids=[r['pid'] for r in rows])
    try:(os.killpg if send is None else send)(group,signal.SIGTERM)
    except ProcessLookupError:return dict(decision='group-already-exited')
    except OSError as error:return dict(decision='refused-signal-error',error=repr(error))
    return dict(decision='signaled-verified-owned-remainder',pids=[r['pid'] for r in rows],qualification='Signal sent,not proof all members have exited')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);a=p.parse_args()
    assert sys.version_info[:2]==(3,12),'Use pinned Python3.12 interpreter'
    root=a.repository.resolve();out=a.output.resolve();assert out.is_relative_to(root) and out!=root;os.chdir(root)
    sys.path.insert(0,str(root/'scripts'));from research import Ledger,read,digest,write_new,review,export
    sys.path.insert(0,str(root/'experiments'/P/'code'));base=load('parent_controller',root/'experiments'/P/'code/run_calibration.py');pruned=load('frozen_pruned',root/'experiments'/P/'code/pruned_expected.py')
    re=load('range_expected',root/'experiments/2026-11-15-range-worker-sweep/code/expected.py')
    inputs=read(a.inputs);receipt=read(root/'deployment-receipt.json');assert receipt['source_commit']==inputs['source_commit'];approval(read(a.approval),inputs['source_commit'],root,out)
    assert digest(a.approval)==inputs['approval_sha256'];protocol=read(root/'experiments'/E/'configs/protocol.json');assert protocol['parent_merge']=='66d7e1e9dfd88954eb52505fa939d68519fc9918';validate_protocol(protocol)
    out.mkdir(parents=True,exist_ok=False)
    for n in ['logs','raw','analysis']:(out/n).mkdir()
    L=Ledger(root);completed=[];attempts=[];deadline=time.monotonic()+72*3600
    def status(state,**extra):
        temp=out/'status.pending';temp.write_text(json.dumps(dict(state=state,utc=base.utc(),source_commit=inputs['source_commit'],completed=len(completed),**extra),indent=2));temp.replace(out/'status.json')
    def source_check():
        for n,h in {**receipt['files'],**receipt.get('support_files',{})}.items():assert digest(root/n)==h,n
        for field in ['cpu_build_manifest','gpu_build_manifest']:
            item=inputs[field];assert digest(item['path'])==item['sha256'];v=read(item['path'])
            for binary in v['binaries'].values():assert digest(binary['path'])==binary['sha256']
            for item in v['logs']:assert digest(item['path'])==item['sha256']
            for item in v.get('generated_sources',[]):assert digest(item['path'])==item['sha256']
        cpu=read(inputs['cpu_build_manifest']['path']);assert cpu['status']=='BUILD_AND_UNIT_TESTS_PASS_NO_MEASUREMENTS'
        materialized=Path(inputs['cpu_build_manifest']['path']).parent/'BMP'
        for n,h in cpu['materialization']['files'].items():assert digest(materialized/n)==h,n
        for name,target in [('guarded_u32.rs','src/project_u32.rs'),('project_benchmark_u32.rs','src/bin/project_benchmark_u32.rs'),('project_benchmark_u64.rs','src/bin/project_benchmark_u64.rs')]:assert digest(root/'experiments'/E/'code'/name)==cpu['materialization']['files'][target]
    def wait(tag):
        while True:
            source_check();before=base.busy_ticks();start=time.monotonic();time.sleep(2);busy=(base.busy_ticks()-before)/os.sysconf('SC_CLK_TCK')/(time.monotonic()-start);gpu=base.gpu_state();memory=base.available_kib()
            row=dict(utc=base.utc(),stage=tag,gpu=gpu,available_kib=memory,other_cpu_cores=busy)
            with (out/'logs/availability.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if not gpu['contexts'] and gpu['free_mib']>=10000 and memory>=34*1024**2 and busy<=4:return
            if time.monotonic()>=deadline:raise RuntimeError('Resource deadline; no foreign process affected')
            status('waiting-for-resources',next_stage=tag);time.sleep(120)
    try:
        source_check();status('validating-full-inputs')
        for field in ['gsp','queries','oracle','query_split','bmp_index','rbp256']:
            item=inputs[field];assert digest(item['path'])==item['sha256'],field
        assert inputs['queries']['sha256']=='150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091'
        assert inputs['gsp']['sha256']=='e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac'
        question=L.event('question',dict(title='Guarded U32 temporary score width in corrected resident BMP',question='How does U32 versus U64 affect matched full-collection calibration latency?',motivation='Known U64 CPU control limitation; no presumed gain or strongestCPU claim',parent_experiment='2026-11-18-matched-final-evaluation',parent_merge=protocol['parent_merge'],authorization_sha256=digest(a.approval),authorization=read(a.approval),scope=protocol['scope']))
        rows=re.queries(Path(inputs['queries']['path']),28679,501,1);assert len(rows)==500;qids=[q for q,_ in rows]
        gold={q:[] for q in qids}
        with open(inputs['oracle']['path']) as f:
            for row in csv.DictReader(f,delimiter='\t'):gold[row['query_id']].append((int(row['original_id']),int(row['score'])))
        assert all(len(v)==10 for v in gold.values())
        exporter=load('pisa_export',root/'experiments/2026-11-18-bmp-full-collection/code/export_queries.py');_,vocab=exporter.vocabulary(Path(inputs['gsp']['path']));assert len(set(vocab))==len(vocab)==28679
        pisa=out/'calibration-500.pisa'
        with pisa.open('x') as f:
            for q,terms in rows:
                assert len(terms)<=128 and len({t for t,_ in terms})==len(terms) and all(1<=w<=32 for _,w in terms)
                f.write(q+': '+' '.join(vocab[t] for t,w in terms for _ in range(w))+'\n')
        assert digest(pisa)=='0ac4ad355c198fffa65a94ec4ae23da19a0fbea5978a690f343e850fdb7a00d9'
        max_terms=max(len(t) for _,t in rows);max_bound=max(255*sum(w for _,w in t) for _,t in rows);assert max_bound<2**24
        cpu=read(inputs['cpu_build_manifest']['path']);gpu=read(inputs['gpu_build_manifest']['path']);assert gpu['source_commit']=='61c8a00d8323d49a011d8909113a8755906323c9'
        metadata=re.rbp_metadata(inputs['rbp256']['path']);assert metadata['docs']==8841823
        # Pinned helper compares all original passage IDs before any scoring.
        wait('verify-original-id-sequence')
        helper=gpu['binaries']['verify_ids'];command=['numactl','--physcpubind=0','--localalloc',helper['path'],inputs['bmp_index']['path'],inputs['gsp']['path']]
        with (out/'logs/full-id-sequence.stdout').open('xb') as so,(out/'logs/full-id-sequence.stderr').open('xb') as se:r=subprocess.run(command,stdout=so,stderr=se,timeout=1800)
        L.event('observation',dict(title='Original-ID helper execution observation',motivation='Preserve helper outcome even on failure',argv=command,returncode=r.returncode,artifacts=[L.blob(out/'logs/full-id-sequence.stdout'),L.blob(out/'logs/full-id-sequence.stderr')]),[question['id']])
        assert r.returncode==0 and read(out/'logs/full-id-sequence.stdout')==dict(documents=8841823,ordered_original_ids_equal=True,gsp_vocab=28679,gsp_nnz=1049282186)
        L.event('observation',dict(title='Full ordered-ID preparation and reusable build verification',motivation='Bind both CPU variants to the same unchanged full collection and frozen GPU anchor',command=command,exit_code=r.returncode,qualification='Uninstrumented ID comparison/build preparation,not retrieval timing',cpu_build=cpu,gpu_build=gpu,artifacts=[L.blob(inputs[k]['path']) for k in ['cpu_build_manifest','gpu_build_manifest']]+[L.blob(item['path']) for build in [cpu,gpu] for item in build['logs']]+[L.blob(Path(inputs['cpu_build_manifest']['path']).parent/'BMP'/n) for n in cpu['materialization']['files']]+[L.blob(item['path']) for item in cpu.get('generated_sources',[])]+[L.blob(item['path']) for build in [cpu,gpu] for item in build['binaries'].values()]),[question['id']])
        write_new(out/'cpu-build.json',cpu);write_new(out/'gpu-build.json',gpu)
        def one(label,tag,repeats,warmups,verify,negative=False):
            wait(tag);directory=out/'raw'/tag;directory.mkdir();is_cpu=label.startswith('bmp');bits=int(label[3:]) if is_cpu else None
            paths={'csv':directory/('samples.tsv' if is_cpu else 'timing.csv'),'top':directory/('results.tsv' if is_cpu else 'top.tsv')}
            if is_cpu:
                paths.update(cpu_monitor=directory/'monitoring.tsv',validation=directory/'validation.json',memory=directory/'memory.json');state=cpu_expected(inputs['bmp_index']['path'],pisa,8841823,28679,1049282186,500,max_terms,max_bound,bits,repeats,warmups)
                binary=cpu['binaries'][str(bits)];argv=[binary['path'],'--index',inputs['bmp_index']['path'],'--queries',pisa,'--expected-docs',8841823,'--expected-vocab',28679,'--expected-nnz',1049282186,'--expected-queries',500,'--cpu',0,'--k',10,'--block-size',32,'--warmups',warmups,'--repeats',repeats,'--samples',paths['csv'],'--results',paths['top'],'--monitoring',paths['cpu_monitor'],'--validation',paths['validation'],'--memory-report',paths['memory']]
                backend_source=cpu['source_commit'];index=inputs['bmp_index']
            else:
                state=pruned.expected(root,inputs['rbp256']['path'],inputs['queries']['path'],metadata,rows,repeats,warmups,verify);binary=gpu['binaries']['prunets'];index=inputs['rbp256'];backend_source=gpu['source_commit']
                argv=[binary['path'],index['path'],inputs['queries']['path'],paths['csv'],paths['top'],500,10,0,verify,repeats,warmups,256,256,inputs['gsp']['path'] if verify else '-',64]
            cfg=dict(state['config']);behavior=state['behavior']
            if negative:assert is_cpu;cfg['score_bits']=64 if bits==32 else 32
            data=[dict(role=role,path=inputs[key]['path'],sha256=inputs[key]['sha256']) for role,key in [('canonical_gsp','gsp'),('queries','queries'),('oracle','oracle'),('query_split','query_split')]]+[dict(role='index',path=index['path'],sha256=index['sha256'])]
            if is_cpu:data.append(dict(role='prepared_queries',path=str(pisa),sha256=digest(pisa)))
            spec=L.plan(dict(title='Guarded CPU width '+tag,question=question['id'],hypothesis='Exact canonical answers; no speed gain presumed',motivation='One-factor CPU temporary width comparison with fixed W256 anchor',owner='vaibhav',expected=cfg,behavior=behavior,split='calibration',success_rule='SPEC_VIOLATION onlyscore_bits before query compute' if negative else 'EXECUTED,allrawanswers exact,eligibletiming whereapplicable',failure_rule='Preserve and stop technical/scientific failure;newtechnicalsubmissionrequiresapproval',next_if_null='Retain validnull/slowerwithouttuning',timing_boundary=BOUNDARY,inputs=data,build=dict(source_commit=backend_source,controller_source_commit=inputs['source_commit'],binary_sha256=binary['sha256'],prepared_build_manifest_sha256=inputs['cpu_build_manifest' if is_cpu else 'gpu_build_manifest']['sha256']),correctness_rule='Independent U64 oracle,complete rawschedule/orderedanswers/fullruntime/physicalwidthguards',analysis_plan=protocol['estimator'],refs=[question['id']]))
            admission=directory/'admission.jsonl';command=[sys.executable,root/'scripts/research.py','run',spec['id'],'--gate-seconds',600]
            for file in [*paths.values(),admission]:command+=['--output',file]
            command+=['--command',sys.executable,root/'experiments'/P/'code/admit_exec.py','--log',admission,*argv];launch=['numactl','--physcpubind=0','--localalloc',*map(str,command)];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUDA_MODULE_LOADING='LAZY');stop=threading.Event();summary={}
            write_new(directory/'runner-command.json',dict(argv=launch,cwd=str(root),environment_overrides={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','RAYON_NUM_THREADS','CUDA_MODULE_LOADING']}))
            with (directory/'runner.stdout').open('xb') as so,(directory/'runner.stderr').open('xb') as se:
                process=subprocess.Popen(launch,cwd=root,env=env,stdout=so,stderr=se,start_new_session=True);observer=None;process_error=None
                try:
                    write_new(directory/'process-identity.json',dict(runner_pid=process.pid,expected_process_group=process.pid,parent_controller_pid=os.getpid(),source_commit=inputs['source_commit'],stage=tag));status('instrumented-run-active',stage=tag,runner_pid=process.pid,experiment=spec['id'])
                    observer=threading.Thread(target=base.monitor,args=(stop,directory/'monitor.jsonl',process.pid,summary,process),daemon=True);observer.start();rc=process.wait(timeout=21600)
                except BaseException as error:
                    process_error=error
                    if process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            process.wait()
                finally:
                    stop.set()
                    if observer is not None and observer.ident is not None:observer.join(timeout=45)
            assert observer is None or not observer.is_alive();write_new(directory/'monitor-summary.json',summary)
            if process_error is not None:
                cleanup=signal_own_remainder(process.pid,directory)
                L.event('observation',dict(title='Owned runner interrupted',stage=tag,actual_returncode=process.returncode,error=repr(process_error),monitor=summary,remaining_group_cleanup=cleanup,qualification='Partial artifacts retained;no invented run_finished'),[spec['id']]);raise process_error
            try:event=runner_result(directory/'runner.stdout')
            except (ValueError,OSError) as error:
                cleanup=signal_own_remainder(process.pid,directory)
                L.event('observation',dict(title='Runner interrupted before final JSON',stage=tag,actual_returncode=rc,monitor=summary,remaining_group_cleanup=cleanup,error=repr(error),qualification='No invented run_finished; retain unmatched start and partial raw'),[spec['id']]);raise RuntimeError('Runner missing final JSON; preserve/inspect') from error
            done=event['id'];payload=event['payload'];records=[json.loads(x) for x in admission.read_text().splitlines()] if admission.exists() else []
            timeout=payload.get('exit_code')==75 and payload.get('runtime') is None and records and records[-1]['decision']=='timeout' and all(not x.exists() for x in paths.values())
            if timeout:valid=False;resource=True;validation=dict(admission_timeout=True);latencies={}
            elif negative:
                valid=payload['status']=='SPEC_VIOLATION' and set(payload.get('runtime_delta') or {})=={'config.score_bits'} and all(x.exists() and len(x.read_text().splitlines())==1 for x in [paths['csv'],paths['top'],paths['cpu_monitor']]) and paths['validation'].exists() and not paths['memory'].exists()
                if valid:cpu_audit(paths['validation'],bits,max_terms,max_bound)
                validation=dict(deliberate_rejection=bool(valid));resource=False;latencies={}
            else:
                valid=rc==0 and payload['status']=='EXECUTED' and payload['runtime']==state and not payload['runtime_delta'];validation={'status':payload['status']};latencies={};resource=False
                if valid:
                    try:
                        latencies,validation=base.validate_outputs('bmp' if is_cpu else label,paths,gold,qids,repeats,warmups,verify,cfg)
                        if is_cpu:
                            audit=cpu_audit(paths['validation'],bits,max_terms,max_bound)
                            memory=read(paths['memory']);assert memory['score_bytes']==bits//8 and memory['process_vm_swap_kib']<=1024**2;validation['memory']=memory
                    except Exception as error:valid=False;validation={'error':repr(error)}
            timing=tag.startswith('timing-')
            if not timeout and (summary.get('errors') or summary.get('own_swap_limit_exceeded') or summary.get('samples',0)<2):valid=False;validation['monitor_technical_failure']=True
            eligible=not summary['foreign_pids'] and not summary['errors'] and summary['samples']>0 and not summary['own_swap_limit_exceeded']
            if timing:eligible=eligible and summary['other_cpu_cores_mean'] is not None and summary['other_cpu_cores_mean']<=4 and (is_cpu or summary['own_gpu_context_seen']) and (not is_cpu or validation.get('iowait_share',1)<=.01)
            resource=resource or bool(valid and timing and not eligible)
            review(L,done,dict(result_summary=dict(validation=validation,monitor=summary,eligible=eligible if timing else None),interpretation='Precompute resource timeout' if timeout else 'Per-stage checks passed' if valid else 'Technical/correctness failure;inspect',conclusions=[dict(text='No comparative claim from this stage',scope='Full8841823collection,500calibration,k10,batch1',evidence_refs=[done])],decision='Resource-only replacement within3attemptcap' if resource else 'Continuefixedschedule' if valid else 'Stop for reviewed repair/newapproval',motivation='Do not tune validslower results or hide technical failures',confidence='high' if valid or timeout else 'low',limitations=['Sampled sharedhost;no exclusive allocation','No heldout access'],next_steps=[dict(task='Complete fixed study and independent review',owner='unassigned',reason='All9timings and allcorrectness required',done_when='Fullprotocolaccepted')]))
            entry=dict(tag=tag,label=label,run=done,raw_directory=str(directory),valid=bool(valid),timing_eligible=bool(eligible) if timing else None,validation=validation,monitor=summary);attempts.append(entry);write_new(out/'analysis'/(tag+'.json'),entry)
            if resource:raise base.ResourceIneligible(tag)
            assert valid,(tag,validation)
            if timing:entry['process_block']=int(tag.split('-')[1][1:])
            completed.append(entry)
            write_new(out/'analysis'/('checkpoint-'+str(len(completed)).zfill(2)+'.json'),dict(source_commit=inputs['source_commit'],completed=completed,attempts=attempts,input_hashes={k:inputs[k]['sha256'] for k in ['gsp','queries','oracle','query_split','bmp_index','rbp256']},qualification='Immutable progress evidence only;no automatictechnicalresubmission orimplicitresume authorization'))
            return entry
        for label in ORDER[:2]:base.resource_retries(one,label,'reject-'+label,1,0,0,True)
        for label in ORDER:base.resource_retries(one,label,'correctness-'+label,1,0,1)
        for block,label in schedule():base.resource_retries(one,label,f'timing-p{block}-'+label,20,3,0)['process_block']=block
        write_new(out/'manifest.json',dict(status='calibration-complete-independent-analysis-pending',source_commit=inputs['source_commit'],cpu_backend_source_commit=cpu['source_commit'],gpu_backend_source_commit=gpu['source_commit'],parent_merge=protocol['parent_merge'],completed=completed,attempts=attempts,inputs=inputs,protocol=protocol,qualification='No comparative conclusion until independent raw/provenance/resource/statistics review;noheldout'))
        write_new(out/'ledger-verify.json',L.verify());export(L);status('calibration-complete-independent-analysis-pending')
    except BaseException as error:
        L.event('observation',dict(title='Guarded CPU calibration stopped',motivation='Preserve failure and partial outputs;no unchanged technical retry',error=repr(error),completed=len(completed),next='Review evidence and seek new technical submission approval'));status('failed',error=repr(error));traceback.print_exc();raise
if __name__=='__main__':main()
