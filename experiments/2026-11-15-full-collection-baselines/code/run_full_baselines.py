"""Instrumented full-collection correctness-first, paired GPU comparison.

Uses the existing project ledger/runner. Each run is registered from resolved
inputs/build/config before the real gate releases query computation.
"""
import argparse,csv,datetime,importlib.util,json,math,os,platform,random,shutil,signal,statistics,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
import numpy as np
from expected_baselines import inherited,csr
from validate_baseline_outputs import validate
from compare_exact import compare_certified
from analyze_full_baselines import analyze

E='2026-11-15-full-collection-baselines'
BOUNDARY='Host construction of the implementation query representation through complete scoring/selection/merging, returned original numeric IDs/scores, result transfer and synchronization. Excludes index build/load/upload, runtime gate, file I/O and correctness validation. Device-span diagnostics have implementation-specific event placement and are not compared.'

def retryable_contended_oom(payload,summary,stderr,allocation,total_bytes,outputs_present,label,docs):
    """Recognize only a failed pre-gate full-input OOM with resource evidence.

    This is a contaminated resource attempt, never an isolated fit/OOM finding.
    A clean-device OOM or a different failure must stop for separate review.
    """
    if label not in ['range','csr','mono'] or docs!=8841823 or outputs_present:return False
    if payload.get('status')!='SPEC_VIOLATION' or payload.get('runtime') is not None:return False
    if payload.get('exit_code',0)==0 or payload.get('error')!='Missing runtime report; compute is unverified':return False
    if not summary.get('external_pids') or summary.get('sampling_errors') or not summary.get('samples') or not summary.get('owned_context_seen'):return False
    # Subtract our maximum sampled usage conservatively; sampled board/context
    # values are evidence of interference, not exact simultaneous physical peaks.
    non_own_estimate=max(0,summary.get('sampled_board_peak_mib',0)-summary.get('sampled_own_peak_mib',0))*1024**2
    if non_own_estimate+allocation<=total_bytes:return False
    if label in ['range','csr']:return stderr.strip()=='ERROR: out of memory'
    last=stderr.strip().splitlines()[-1] if stderr.strip() else ''
    return last.startswith('torch.OutOfMemoryError: CUDA out of memory.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-commit',required=True);a=parser.parse_args()
    root=a.repository.resolve();out=a.output.resolve();os.chdir(root);sys.path.insert(0,str(root/'scripts'))
    from research import Ledger,digest,read,write_new,review,compare,differences,export
    if not out.is_relative_to(root):raise ValueError('output must be inside deployed repository')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['logs','raw','analysis','jit','cuda-cache','checkpoints','build']: (out/name).mkdir()
    l=Ledger(root);inputs=read(a.inputs);receipt=read(root/'deployment-receipt.json');protocol=read(root/'experiments'/E/'configs/protocol.json')
    qrows=[];completed=[];builds={};question=None;frozen=None;current=None;known_events=set();known_blobs=set();known_files=set();availability_count=[0];target=inputs['target_uuid']
    allowed_cpus=sorted(os.sched_getaffinity(0));cpu=allowed_cpus[0];monitor_cpu=allowed_cpus[1] if len(allowed_cpus)>1 else cpu
    def status(state,stage,**kw):
        temp=out/'status.pending';temp.write_text(json.dumps(dict(state=state,stage=stage,source_commit=a.source_commit,pid=os.getpid(),completed_runs=len(completed),updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));temp.replace(out/'status.json')
    def verify_source():
        if receipt['source_commit']!=a.source_commit:raise ValueError('deployed commit mismatch')
        for name,h in {**receipt['files'],**receipt.get('support_files',{})}.items():
            if not (root/name).resolve().is_relative_to(root) or digest(root/name)!=h:raise ValueError('source/support changed: '+name)
    def process(tag,argv,env=None):
        write_new(out/'logs'/(tag+'-command.json'),dict(argv=list(map(str,argv)),cwd=str(root),source_commit=a.source_commit,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),qualification='Uninstrumented build/import/probe observation.'))
        with (out/'logs'/(tag+'.stdout.log')).open('xb') as stdout,(out/'logs'/(tag+'.stderr.log')).open('xb') as stderr:
            p=subprocess.run(list(map(str,argv)),cwd=root,stdout=stdout,stderr=stderr,env=env)
        if p.returncode:raise RuntimeError(tag+' failed; inspect saved logs')
        write_new(out/'logs'/(tag+'-exit.json'),dict(returncode=p.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    def gpu_state():
        p=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,name,compute_cap,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if p.returncode:raise RuntimeError('GPU state unavailable')
        row=next(csv.reader(p.stdout.splitlines()));row=[s.strip() for s in row]
        if row[0]!=target or row[1]!='NVIDIA GeForce RTX 4070' or row[2]!='8.9':raise ValueError('target GPU changed')
        p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if p.returncode:raise RuntimeError('compute-context state unavailable')
        contexts=[r for r in csv.reader(p.stdout.splitlines()) if r and r[0].strip()==target]
        return dict(uuid=row[0],name=row[1],compute_capability=row[2],total_bytes=int(row[3])*1024**2,free_bytes=int(row[4])*1024**2,contexts=contexts)
    def wait_idle(required=0):
        deadline=time.monotonic()+protocol['availability_wait_limit_seconds']
        while True:
            state=gpu_state()
            availability_count[0]+=1
            write_new(out/'logs'/('availability-%06d.json'%availability_count[0]),dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),required_bytes=required,**state))
            if not state['contexts'] and state['free_bytes']>=required:return state
            if time.monotonic()>=deadline:raise RuntimeError('availability wait limit reached; no benchmark launched on occupied GPU')
            status('waiting','target-GPU-idle-and-memory',required_bytes=required,observed_free_bytes=state['free_bytes'],compute_context_count=len(state['contexts']))
            time.sleep(600)
    def sampler(stop,path,pgid,summary):
        os.sched_setaffinity(0,{monitor_cpu});seen={};frames=0;external=set();errors=[];own_peak=0;board_peak=0;owned_seen=False
        with path.open('x') as f:
            while not stop.is_set():
                frame={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                try:
                    p=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,memory.used,memory.free,clocks.sm,pstate,power.draw,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                    if p.returncode:raise RuntimeError('GPU sampling exit '+str(p.returncode))
                    row=[x.strip() for x in next(csv.reader(p.stdout.splitlines()))]
                    if row[0]!=target:raise ValueError('sampled GPU changed')
                    frame['gpu']=row;board_peak=max(board_peak,int(row[1]))
                    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                    if p.returncode:raise RuntimeError('process sampling exit '+str(p.returncode))
                    contexts=[];owned_memory=0
                    for row in csv.reader(p.stdout.splitlines()):
                        if not row or row[0].strip()!=target:continue
                        pid=int(row[1]);memory=int(row[2])
                        try:group=os.getpgid(pid);seen[pid]=group
                        except ProcessLookupError:group=seen.get(pid)
                        if group is None:raise RuntimeError('unresolved sampled compute process')
                        owned=group==pgid;owned_seen|=owned
                        if owned:owned_memory+=memory
                        else:external.add(pid)
                        contexts.append(dict(pid=pid,pgid=group,used_mib=memory,owned=owned))
                    own_peak=max(own_peak,owned_memory);frame['contexts']=contexts;frames+=1
                except Exception as error:frame['error']=type(error).__name__+': '+str(error);errors.append(frame['error'])
                f.write(json.dumps(frame)+'\n');f.flush();stop.wait(1)
        summary.update(samples=frames,owned_context_seen=owned_seen,external_pids=sorted(external),sampling_errors=errors,sampled_own_peak_mib=own_peak,sampled_board_peak_mib=board_peak,qualification='One-second sampled peaks, not exact physical peaks; no sampled foreign context is not an exclusivity proof.')
    def checkpoint(tag):
        events=l.events();fresh=[e for e in events if e['id'] not in known_events];paths={}
        for event in fresh:
            p=Path('research/events')/(event['id']+'.json');paths[str(p)]=root/p
            for blob in event['payload'].get('artifacts',[]):
                if blob['sha256'] not in known_blobs:paths[blob['blob']]=root/blob['blob']
        for p in (root/'research/evidence').glob('*'):
            if p.is_file() and p.name not in known_blobs:paths[str(p.relative_to(root))]=p
        for p in out.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.relative_to(out).parts and p.name!='status.json' and str(p.relative_to(root)) not in known_files and p.stat().st_size<64*1024**2:paths[str(p.relative_to(root))]=p
        starts={e['id'] for e in events if e['kind']=='run_started'};finished={ref for e in events if e['kind']=='run_finished' for ref in e['refs']}
        for identity in starts-finished:
            for p in (root/'research/runs'/identity).glob('*'):
                if p.is_file():paths[str(p.relative_to(root))]=p
        archive=out/'checkpoints'/(tag+'.zip');pending=archive.with_suffix('.pending')
        with zipfile.ZipFile(pending,'x',zipfile.ZIP_DEFLATED) as z:
            for name,p in sorted(paths.items()):z.write(p,name)
            z.writestr(str(out.relative_to(root)/'private-support/status-snapshots'/(tag+'.json')),(out/'status.json').read_bytes())
        with zipfile.ZipFile(pending) as z:
            if z.testzip() is not None:raise ValueError('checkpoint CRC')
        with pending.open('rb') as f:os.fsync(f.fileno())
        pending.rename(archive)
        write_new(out/'checkpoints'/(tag+'.json'),dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,new_events=len(fresh),qualification='Scratch incremental ledger objects plus current small run files; bulk datasets excluded. Local transfer still required.'))
        known_events.update(e['id'] for e in fresh)
        known_blobs.update(blob['sha256'] for e in fresh for blob in e['payload'].get('artifacts',[]))
        known_blobs.update(Path(name).name for name in paths if name.startswith('research/evidence/'))
        known_files.update(paths)
    def review_run(done,result,interpretation,decision,artifacts=()):
        return review(l,done['id'],dict(result_summary=result,interpretation=interpretation,conclusions=[dict(text='Declared run checks only; no comparative performance claim at this stage.',scope='Full calibration or named synthetic/engineering input as registered.',evidence_refs=[done['id']])],decision=decision,motivation='Preserve attempt, exactness, qualification and next action.',confidence='high',limitations=['Shared GPU; sampled context absence is not exclusivity.','No held-out, outsideCPU-BMP, unmodified nativeGPUSparse or serving-tail claim.','C++ tracked buffers and Torch allocator peaks are not physical board peaks.'],next_steps=[dict(task=decision,owner='unassigned',reason='Required correctness/performance protocol.',done_when='Saved evidence passes declared checks.')],refs=[frozen['id']],artifacts=list(artifacts)))
    status('running','source-and-input-verification')
    def on_signal(number,frame):raise RuntimeError('Controller signal '+str(number))
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    try:
        if cpu!=0:raise ValueError('Frozen CPU0 unavailable; register a revised protocol before execution')
        os.sched_setaffinity(0,{cpu})
        verify_source();l.verify()
        if platform.node().split('.')[0]!='cuda5':raise ValueError('wrong host')
        for role in ['gsp','rbp','queries','query_split','oracle']:
            if inputs[role]['sha256']!=protocol['inputs'][role]['sha256']:raise ValueError('input differs from frozen protocol: '+role)
            if digest(inputs[role]['path'])!=inputs[role]['sha256']:raise ValueError('input hash: '+role)
        if target!=protocol['target_uuid'] or protocol['docs']!=8841823 or protocol['query_count']!=500 or protocol['k']!=10 or protocol['batch_size']!=1 or protocol['warmup_rounds']!=3 or protocol['measured_rounds']!=20:raise ValueError('unfrozen operating point')
        re,me=inherited(root);meta=re.rbp_metadata(inputs['rbp']['path']);proof=l.get(inputs['index_validation_ref'])['payload']['result']
        if not proof['success'] or proof['docs']!=8841823 or proof['maximum_term_document_multiplicity']!=1:raise ValueError('missing full validation')
        gsp=dict(docs=proof['docs'],vocab=proof['vocab'],nnz=proof['nnz'],multiplicity=1,sha256=inputs['gsp']['sha256'])
        qrows=re.queries(inputs['queries']['path'],gsp['vocab'],limit=500,multiplicity=1)
        if len(qrows)!=500:raise ValueError('calibration query count: expected500, got%d'%len(qrows))
        if meta['source_sha256']!=gsp['sha256']:raise ValueError('packed index source GSP binding')
        question=l.event('question',dict(title='Full8841823-passage matched exact GPU baselines',motivation='Determine full-collection latency/resource ordering rather than extrapolating1M results.',source_commit=a.source_commit,parent_merge=receipt['parent_merge'],artifacts=[l.blob(root/'experiments'/E/'configs/protocol.json'),l.blob(a.inputs),l.blob(root/'deployment-receipt.json')]+[l.blob(p) for p in (root/'experiments'/E/'code').glob('*.py')]),[inputs['oracle']['run'],inputs['index_validation_ref']])
        frozen=l.event('decision',dict(title='Freeze full-calibration comparison protocol before GPU execution',motivation='Fixed operating point, schedule, exactness and resource rules.',protocol=protocol,qualification='Calibration protocol; not a held-out freeze.',refs=[question['id']]),[question['id']])
        status('running','compile-reviewed-CUDA-sources')
        process('nvcc-version',['nvcc','--version'])
        for label,source in [('range','cuda_range_v2.cu'),('csr','cuda_fullbound.cu')]:
            binary=out/'build'/('cuda_'+label);argv=['nvcc','-O3','-lineinfo','-arch=sm_89','-std=c++17',root/'src'/source,'-o',binary];process('build-'+label,argv)
            builds[label]=dict(source_commit=a.source_commit,binary_sha256=digest(binary),source_sha256=digest(root/'src'/source),runtime_sha256=digest(root/'src/research_runtime.h'),flags=['-O3','-lineinfo','-arch=sm_89','-std=c++17'],compiler=(out/'logs/nvcc-version.stdout.log').read_text(),target_uuid=target)
        mono=root/'experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64/canonical_baseline.py';native=inputs['native_repo'];python=sys.executable
        probe="import json,numpy,scipy,torch,triton;print(json.dumps(dict(torch_version=str(torch.__version__),triton_version=triton.__version__,numpy_version=numpy.__version__,scipy_version=scipy.__version__,torch_cuda_version=torch.version.cuda)))"
        process('packages',[python,'-c',probe]);environment=json.loads((out/'logs/packages.stdout.log').read_text());environment.update(gpu_name='NVIDIA GeForce RTX 4070',compute_capability='8.9')
        builds['mono']=dict(source_commit=a.source_commit,upstream_revision=inputs['native_revision'],adapter_hashes={p.name:digest(p) for p in mono.parent.glob('*.py')},packages=environment,target_uuid=target,block_pl=512)
        build_files=[p for p in (out/'logs').iterdir() if p.is_file()]+list(mono.parent.glob('*.py'))+[root/'experiments/2026-11-15-range-worker-sweep/code/expected.py',Path(native)/'src/triton_kernel.py']
        build_event=l.event('observation',dict(title='Exact full-baseline build and environment',motivation='Record actual binaries and adapted source before retrieval.',builds=builds,artifacts=[l.blob(p) for p in build_files]),[question['id']])
        frozen_builds={k:dict(v,build_record=build_event['id']) for k,v in builds.items()}
        checkpoint('build-prepared')
        def one(tag,label,index,query,oracle,gm,rm,rows,split,repeats,warmups,verify,negative=None):
            nonlocal current
            verify_source();directory=out/'raw'/tag;directory.mkdir(exist_ok=False)
            csvfile=directory/'timing.csv';tops=directory/'top.tsv';native_report=directory/'native-validation.json';jit=directory/'jit-manifest.json'
            if label=='range':
                cfg=re.config(index,query,rm,gm,rows,10,256,repeats,warmups,verify,cpu);behavior=re.marks();allocation=re.allocation(rm,10,256)
                argv=[out/'build/cuda_range',index,query,csvfile,tops,len(rows),10,0,verify,repeats,warmups,256,256,gm['path'] if verify else '-']
            elif label=='csr':
                cfg,behavior,allocation=csr(index,query,gm,rows,10,repeats,warmups,verify,cpu,read(root/'research/policy.json')['sticky_behavior'])
                argv=[out/'build/cuda_csr',index,query,csvfile,tops,len(rows),10,0,verify,32,0,repeats,warmups]
            else:
                cfg=me.expected(index,query,oracle,native,mono,environment,cpu,len(rows),10,512,repeats,warmups,verify,{'triton_kernel.py':inputs['native_monolithic_sha256']},inputs['native_revision']);behavior=me.marks(verify);allocation=cfg['index_gpu_bytes']+cfg['dense_score_bytes']+cfg['packed_key_bytes']
                argv=[python,mono,'--variant','monolithic','--index',index,'--queries',query,'--reference',oracle,'--native-repo',native,'--csv',csvfile,'--top',tops,'--validation',native_report,'--jit-manifest',jit,'--limit',len(rows),'--k',10,'--gpu',0,'--block-pl',512,'--repeats',repeats,'--warmups',warmups,'--verify',verify]
            actual_cfg=dict(cfg)
            if cfg['docs']!=gm['docs'] or cfg['query_count']!=len(rows):raise ValueError('resolved config/input scope differs')
            if label=='mono' and (cfg['nnz']!=gm['nnz'] or cfg['vocab']!=gm['vocab']):raise ValueError('native header changed after qualification')
            if negative:cfg[negative]+=1
            query_sha=inputs['queries']['sha256'] if split=='calibration' else digest(query)
            oracle_sha=inputs['oracle']['sha256'] if split=='calibration' else digest(oracle)
            items=[dict(role='canonical_gsp',path=str(gm['path']),sha256=gm['sha256']),dict(role='queries',path=str(query),sha256=query_sha),dict(role='reference',path=str(oracle),sha256=oracle_sha)]
            if label=='range':items.append(dict(role='index',path=str(index),sha256=rm['file_sha256']))
            if split=='calibration':items.append(dict(role='query_split',path=inputs['query_split']['path'],sha256=inputs['query_split']['sha256']))
            spec=l.plan(dict(title='Full baseline '+tag,question=question['id'],hypothesis='Exact common outputs; empirical full-corpus speed/resource ordering.' if not negative else 'Actual runtime drift is rejected before query computation.',motivation='Frozen full-baseline protocol with independent canonical checks.',owner='vaibhav',expected=cfg,behavior=behavior,split=split,success_rule='Deliberate drift: SPEC_VIOLATION and no query outputs.' if negative else 'Executed accepted real runtime; complete exact canonical outputs and all declared samples.',failure_rule='Preserve failure; inspect and separately review repair<=3. No repeated known-broken code.',next_if_null='Retain valid slower results; no outcome-driven retuning.',timing_boundary='Engineering runtime rejection only; no query compute.' if negative else BOUNDARY,correctness_rule=protocol['correctness'],analysis_plan=protocol['analysis'],inputs=items,build=frozen_builds[label],parent_external_experiment='2026-11-15-range-worker-sweep',declared_changes=protocol['declared_changes'],refs=[frozen['id'],build_event['id']]))
            required=allocation+(1536 if label=='mono' else 1024)*1024**2;available=wait_idle(required);status('running',tag,label=label)
            command=[python,root/'scripts/research.py','run',spec['id'],'--gate-seconds','1800']
            for path in [csvfile,tops]+([native_report,jit] if label=='mono' else []):command+=['--output',path]
            command+=['--command',*argv];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',CUDA_MODULE_LOADING='LAZY',TRITON_CACHE_DIR=str(out/'jit'/tag),CUDA_CACHE_PATH=str(out/'cuda-cache'/tag))
            stop=threading.Event();summary={};stdout=out/'logs'/(tag+'.runner.stdout.log');stderr=out/'logs'/(tag+'.runner.stderr.log')
            with stdout.open('xb') as so,stderr.open('xb') as se:
                p=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=so,stderr=se,start_new_session=True)
                observer=threading.Thread(target=sampler,args=(stop,out/'logs'/(tag+'.monitor.jsonl'),p.pid,summary),daemon=True);observer.start()
                try:code=p.wait(timeout=7200)
                except BaseException:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise
                finally:stop.set();observer.join(timeout=40)
            if observer.is_alive():raise RuntimeError('monitor did not finish')
            write_new(out/'analysis'/(tag+'-monitor.json'),summary)
            current=read(stdout)
            if negative:
                if current['payload']['status']!='SPEC_VIOLATION' or any(p.exists() for p in [csvfile,tops]+([native_report,jit] if label=='mono' else [])):raise ValueError('deliberate runtime drift was not safely rejected')
                if set(current['payload']['runtime_delta'])!={'config.'+negative} or current['payload']['runtime']['config'].get(negative)!=actual_cfg[negative]:raise ValueError('wrong runtime rejection reason or missing actual setting')
                review_run(current,dict(expected_rejection=True,field=negative),'Runtime mismatch rejected before query outputs.','Continue frozen protocol');completed.append(dict(tag=tag,label=label,run=current['id'],kind='engineering'));checkpoint(tag);return None
            if code or current['payload']['status']!='EXECUTED':
                errors=[artifact for artifact in current['payload'].get('artifacts',[]) if artifact['name']=='stderr.txt']
                child_error=(root/errors[0]['blob']).read_text(errors='replace') if len(errors)==1 else ''
                outputs_present=any(path.exists() for path in [csvfile,tops]+([native_report,jit] if label=='mono' else []))
                resource=retryable_contended_oom(current['payload'],summary,child_error,allocation,available['total_bytes'],outputs_present,label,gm['docs'])
                if resource:
                    entry=dict(tag=tag,label=label,run=current['id'],kind='resource_failed',eligible=False,monitor=summary,classification='Contaminated pre-gate OOM; no isolated memory-fit or performance conclusion.')
                    path=out/'analysis'/(tag+'-resource-failure.json');write_new(path,dict(**entry,modeled_application_or_core_bytes=allocation,total_device_bytes=available['total_bytes'],outputs_present=outputs_present,decision='Retain failed attempt and retry only after target GPU becomes idle, within3fresh resource replacements.'))
                    review_run(current,dict(status=current['payload']['status'],monitor=summary,resource_failure=True),'Out of memory before runtime report with sampled foreign context and insufficient modeled remaining capacity. No retrieval completion, exactness, timing or isolated OOM claim.','Wait for idle GPU and use fresh registered resource replacement',[l.blob(path)])
                    completed.append(entry);checkpoint(tag);return entry
                review_run(current,dict(status=current['payload']['status'],monitor=summary),'Attempt incomplete; no exactness/performance claim.','Inspect before reviewed repair');checkpoint(tag);raise RuntimeError(tag+' not executed')
            if digest(oracle)!=oracle_sha or digest(query)!=query_sha:raise ValueError('reference/query changed before validation')
            report,latencies=validate(label,csvfile,tops,oracle,gm['path'],rows,actual_cfg,allocation,current['id'],oracle_sha,query_sha,gm['sha256'],native_report if label=='mono' else None)
            path=out/'analysis'/(tag+'-validation.json');write_new(path,report)
            validation_blob=l.blob(path);monitor_blob=l.blob(out/'analysis'/(tag+'-monitor.json'))
            timing=tag.startswith('timing-')
            eligible=(not summary['external_pids'] and not summary['sampling_errors'] and summary['samples']>0 and summary['owned_context_seen']) if timing else True
            review_run(current,dict(validation=report,monitor=summary,timing_eligible=eligible if timing else None),'Canonical outputs passed; timing eligibility is independent of latency. Small correctness processes may end between monitor samples and do not certify latency.','Continue fixed protocol' if eligible else 'Retain ineligible run; inspect resource/monitor condition before fresh replacement',[validation_blob,monitor_blob])
            entry=dict(tag=tag,label=label,run=current['id'],csv=str(csvfile),top=str(tops),validation=validation_blob,monitor=summary,eligible=eligible,latencies=latencies,build=frozen_builds[label]);completed.append(entry);write_new(out/'analysis'/(tag+'-completed.json'),{k:v for k,v in entry.items() if k!='latencies'});checkpoint(tag)
            if timing and (summary['sampling_errors'] or not summary['owned_context_seen']):raise RuntimeError('Missing/failed monitor evidence')
            return entry
        def stable(tag,*args,**kw):
            for attempt in range(4):
                entry=one(tag+('-resource%d'%attempt if attempt else ''),*args,**kw)
                if entry is None or entry['eligible']:return entry
            raise RuntimeError('Persistent sampled contention after3fresh resource replacements')
        # Definitions are inherited, not regenerated with guessed defaults.
        factory=module_from_path('full_parent_fixtures',root/'tests/check_range_bitpack.py');rng=random.Random(20261114);cases=[]
        folder=out/'fixtures';folder.mkdir()
        for label,(ids,docs,vocab) in factory.fixtures().items():
            if label=='duplicate':continue # Unsupported by the pinned native loader; full corpus proves multiplicity1.
            case=folder/label;case.mkdir();g=case/'index.gsp';q=case/'queries.tsv';o=case/'oracle.json';r=case/'index.rbp'
            factory.write_gsp(g,ids,docs,vocab);queries=factory.queries_for(label,vocab,rng)
            with q.open('x') as f:
                for qid,weights in queries.items():f.write(qid+''.join(' %d:%d'%pair for pair in weights.items())+'\n')
            write_new(o,{qid:factory.expected_ranking(ids,docs,weights) for qid,weights in queries.items()});cases.append((label,g,q,o,r))
        from prepare_reference import make_fixture
        case=folder/'numeric-uint64';case.mkdir();g,q,oracle=make_fixture(case);o=case/'oracle.json';write_new(o,oracle);cases.append(('numeric-uint64',g,q,o,case/'index.rbp'))
        tool=Path(inputs['range_tool']['path'])
        if digest(tool)!=inputs['range_tool']['sha256']:raise ValueError('reused index tool hash')
        for name,h in inputs['range_tool']['source_files'].items():
            if digest(root/name)!=h:raise ValueError('reused index builder source differs')
        full_gm=dict(gsp,path=inputs['gsp']['path']);correctness=[];timings=[]
        for label,g,q,o,r in cases:
            process('fixture-index-'+label,[tool,'build',g,r,1024,'positions']);gm=re.gsp_metadata(g);gm['path']=str(g);rm=re.rbp_metadata(r);rows=re.queries(q,gm['vocab'],limit=100000,multiplicity=gm['multiplicity'])
            l.event('observation',dict(title='Packed synthetic fixture '+label,motivation='Preserve exact independent input and uninstrumented preparation evidence.',tool=inputs['range_tool'],argv=list(map(str,[tool,'build',g,r,1024,'positions'])),artifacts=[l.blob(p) for p in [g,q,o,r,out/'logs'/('fixture-index-'+label+'.stdout.log'),out/'logs'/('fixture-index-'+label+'.stderr.log')]]),[question['id']])
            for kind in ['range','mono','csr']:stable('fixture-'+label+'-'+kind,kind,r if kind=='range' else g,q,o,gm,rm,rows,'fixture',1,0,1)
        # All three adapters must demonstrate the actual external gate.
        label,g,q,o,r=cases[1];gm=re.gsp_metadata(g);gm['path']=str(g);rm=re.rbp_metadata(r);rows=re.queries(q,gm['vocab'],limit=100000,multiplicity=gm['multiplicity'])
        for kind,field in [('range','score_threads'),('mono','query_count'),('csr','batch_size')]:one('reject-'+kind,kind,r if kind=='range' else g,q,o,gm,rm,rows,'fixture',1,0,1,negative=field)
        for kind in ['range','mono','csr']:
            entry=stable('correctness-'+kind,kind,inputs['rbp']['path'] if kind=='range' else inputs['gsp']['path'],inputs['queries']['path'],inputs['oracle']['path'],full_gm,meta,qrows,'calibration',1,0,0);correctness.append(entry)
        maximum_sum=max(sum(w for t,w in row) for q,row in qrows)
        artifacts=[entry['validation'] for entry in correctness]
        cert=l.event('observation',dict(title='Certified finite exact U32/U64 full-calibration domain',motivation='Retain physical arithmetic labels while proving exact common results on frozen inputs.',certificate_kind='guarded-u32-u64-exact-domain-v1',docs=8841823,query_count=500,k=10,batch_size=1,canonical_gsp_sha256=inputs['gsp']['sha256'],queries_sha256=inputs['queries']['sha256'],index_validation_ref=inputs['index_validation_ref'],maximum_term_document_multiplicity=1,max_query_weight_sum=maximum_sum,max_conservative_score_bound=255*maximum_sum,max_query_terms=max(len(row) for q,row in qrows),unique_cap32_query_terms_verified=True,oracle_run=inputs['oracle']['run'],oracle_sha256=inputs['oracle']['sha256'],correctness=[dict(run=entry['run'],validation=entry['validation']) for entry in correctness],artifacts=artifacts),[inputs['index_validation_ref'],inputs['oracle']['run']]+[entry['run'] for entry in correctness])
        for repetition,order in enumerate(protocol['process_order']):
            for kind in order:
                entry=stable('timing-r%d-%s'%(repetition,kind),kind,inputs['rbp']['path'] if kind=='range' else inputs['gsp']['path'],inputs['queries']['path'],inputs['oracle']['path'],full_gm,meta,qrows,'calibration',20,3,0);entry['repetition']=repetition;timings.append(entry)
        status('running','paired-analysis')
        analysis=analyze(timings,[qid for qid,row in qrows],protocol['analysis']['bootstrap_seed']);write_new(out/'analysis/paired-results.json',analysis)
        comparisons=[]
        for repetition in range(3):
            pairs={entry['label']:entry for entry in timings if entry['repetition']==repetition}
            for label in ['mono','csr']:
                left,right=pairs['range']['run'],pairs[label]['run'];probe=compare(l,left,right,{},'Inspect all system-comparison deltas before domain-qualified comparison.')
                allowed={key:comparison_reason(key,probe['payload']['delta'][key]) for key in probe['payload']['delta']}
                comparison=compare_certified(l,left,right,allowed,'Frozen full-corpus system comparison; inputs/outputs and operating point matched, implementation/resource/representation differences disclosed.',cert['id'],target);comparisons.append(comparison['id'])
        decision=l.event('decision',dict(title='Complete full-calibration matched baseline comparison',motivation='Empirical full-corpus evidence now supports scoped conclusions; retain unsupported resident pruning modes.',result=analysis,comparisons=comparisons,domain_certificate=cert['id'],unsupported=protocol['unsupported'],limitations=protocol['limitations'],next_action='Independent local raw replay/provenance/statistics validation before publication. Only then choose the reviewed selection-capacity follow-on.',artifacts=[l.blob(out/'analysis/paired-results.json')]),comparisons+[cert['id']])
        write_new(out/'manifest.json',dict(source_commit=a.source_commit,parent_merge=receipt['parent_merge'],protocol=frozen['id'],decision=decision['id'],domain_certificate=cert['id'],comparisons=comparisons,completed=[{k:v for k,v in entry.items() if k not in ['latencies','build']} for entry in completed],analysis=analysis))
        write_new(out/'ledger-verify.json',l.verify());export(l);checkpoint('final');status('complete','finished',decision=decision['id'])
    except BaseException as error:
        events=l.events();reviewed={e['payload'].get('run') for e in events if e['kind']=='review'};pending=[e['id'] for e in events if e['kind']=='run_finished' and e['id'] not in reviewed]
        starts={e['id'] for e in events if e['kind']=='run_started'};finished={ref for e in events if e['kind']=='run_finished' for ref in e['refs']};unfinished=sorted(starts-finished)
        payload=dict(title='Full-baseline controller stopped',motivation='Preserve failure and exact unfinished work before any reviewed repair.',error={'type':type(error).__name__,'message':str(error)},current_run=current['id'] if current else None,runs_pending_review=pending,unfinished_run_starts=unfinished,completed_runs=len(completed),next_action='Inspect logs/source/config/output; review before up to3technical repairs. Do not rerun unchanged known-broken code or tune a valid negative outcome.')
        l.event('observation',payload,[question['id']] if question else [])
        if pending:l.event('task',dict(title='Review stopped full-baseline attempts',motivation='No failed or interrupted attempt may be silently skipped.',runs=pending,next_action='Inspect raw outputs/configs/errors and append reviews before repair.'),pending)
        if unfinished:l.event('task',dict(title='Resolve interrupted full-baseline attempts',motivation='An unmatched start remains INTERRUPTED/UNKNOWN, not a completed or successful run.',runs=unfinished,next_action='Inspect preserved runner directory/partial outputs and append correction/review before reuse.'),unfinished)
        try:checkpoint('stopped')
        except Exception as archive_error:payload['checkpoint_error']=str(archive_error)
        status('failed','stopped',error=payload['error']);traceback.print_exc();raise

def module_from_path(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def comparison_reason(key,delta):
    if key.startswith('behavior.contract') or key.startswith('behavior.arithmetic'):return 'Distinct physical integer accumulators; require explicit finite-domain exactness certificate, never a silent relabel.'
    if key.startswith('environment.'):return 'Same required GPUUUID/model/driver and frozen CPU affinity; process-time clocks/pstate/free memory can vary. Raw sampled context/clocks and balanced process order retained; residual hardware variability limits inference.'
    if key.startswith('inputs.index') or key.startswith('config.index_path'):return 'Different index representation from identical canonical full GSP; hashes/source mapping and every-posting range roundtrip preserved.'
    if key.startswith('config.') or key.startswith('behavior.'):return 'Audited implementation-specific query/index representation, traversal, selection, adapter/runtime reporting; common effective score/ties/inputs/GPU/k/batch/warmups/repeats/boundary checked separately.'
    if key.startswith(('build.','binaries.','source.')):return 'System comparison of explicitly pinned compiled C++ implementation versus adapted Torch/Triton source; build/source/binary/JIT provenance retained.'
    return 'Explicit frozen system-comparison difference: '+json.dumps(delta,sort_keys=True)

if __name__=='__main__':main()
