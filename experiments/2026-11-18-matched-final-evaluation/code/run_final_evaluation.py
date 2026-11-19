"""Submit-time frozen heldout preparation, instrumented U64 oracle, then five systems."""
import argparse,csv,datetime,hashlib,json,os,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
from prepare_heldout_queries import prepare,digest,CAL_SHA
import run_heldout as controller

def main():
 def terminate_owned(signum,frame):raise KeyboardInterrupt('Owned pipeline termination')
 signal.signal(signal.SIGTERM,terminate_owned)
 p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);a=p.parse_args();root=a.repository.resolve();os.chdir(root);sys.path.insert(0,str(root/'scripts'));from research import Ledger,read,write_new,review,export
 L=Ledger(root);inputs=read(root/'support/heldout-base-inputs.json');receipt=read(root/'deployment-receipt.json');frozen=L.get(inputs['method_freeze']);assert frozen['kind']=='freeze';method=frozen['payload']['evaluation_rule'];assert method['query_count']==6480 and method['source_commit']==inputs['source_commit']==receipt['source_commit']
 work=root/'results/2026-11-18-matched-final-evaluation/20261119-cuda5-heldout01-preparation';out=root/'results/2026-11-18-matched-final-evaluation/20261119-cuda5-heldout01';work.mkdir(parents=True,exist_ok=False);(work/'logs').mkdir();deadline=time.monotonic()+259200
 def state(stage,**kw):
  v=dict(state='failed' if stage=='failed' else 'complete' if stage.startswith('complete-') else 'active',stage=stage,utc=controller.utc(),pipeline_pid=os.getpid(),source_commit=inputs['source_commit'],output=str(out),**kw);t=root/'evaluation-status.pending';t.write_text(json.dumps(v,indent=2));t.replace(root/'evaluation-status.json')
 def source_check():
  for n,h in {**receipt['files'],**receipt['support_files']}.items():assert digest(root/n)==h,n
  assert digest(inputs['build_manifest']['path'])==inputs['build_manifest']['sha256']
  for x in read(inputs['build_manifest']['path'])['binaries'].values():assert digest(x['path'])==x['sha256']
 def wait(tag):
  while True:
   source_check();start=controller.busy_ticks();t=time.monotonic();time.sleep(2);cpu=(controller.busy_ticks()-start)/os.sysconf('SC_CLK_TCK')/(time.monotonic()-t);gpu=controller.gpu_state();mem=controller.available_kib()
   with (work/'logs/admission.jsonl').open('a') as f:f.write(json.dumps(dict(utc=controller.utc(),stage=tag,gpu=gpu,available_kib=mem,busy_cores=cpu))+'\n')
   if not gpu['contexts'] and gpu['free_mib']>=10000 and mem>=34*1024**2 and cpu<=4:return
   if time.monotonic()>=deadline:raise RuntimeError('Resource wait deadline; no foreign process affected')
   state('waiting-for-resources',next_stage=tag);time.sleep(120)
 def child(tag,argv,timeout):
  source_check();state(tag,argv=list(map(str,argv)))
  with (work/'logs'/(tag+'.stdout')).open('xb') as so,(work/'logs'/(tag+'.stderr')).open('xb') as se:
   proc=subprocess.Popen(list(map(str,argv)),cwd=root,stdout=so,stderr=se,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',CUDA_MODULE_LOADING='LAZY'))
   state(tag,child_pid=proc.pid)
   try:rc=proc.wait(timeout=timeout)
   except BaseException:
    if proc.poll() is None:
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:
      try:os.killpg(proc.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      proc.wait()
    raise
  assert rc==0,(tag,rc);return rc
 exposed=False
 try:
  source_check();L.verify();wait('before-query-preparation');state('preparing-frozen-heldout-queries')
  exposed=True
  info=prepare(root,method['raw_query_path'],inputs['query_split']['path'],inputs['calibration_queries']['path'],work/'queries',root/'research/events'/(inputs['method_freeze']+'.json'))
  L.event('observation',dict(title='Frozen heldout preprocessing',motivation='Reproduce original500calibration bytes before accepting all6480heldout rows; no tuning or drops',result=info,artifacts=[L.blob(work/'queries/preparation.json')]),[inputs['method_freeze']])
  wait('before-independent-reference-build');binary=work/'cpu_reference';compiler=['g++','-O3','-std=c++17','-pthread',root/'src/cpu_pilot.cpp','-o',binary];child('compiler-version',['g++','--version'],60);child('reference-build',compiler,600)
  build={'source_commit':inputs['source_commit'],'source_sha256':digest(root/'src/cpu_pilot.cpp'),'runtime_sha256':digest(root/'src/research_runtime.h'),'binary_sha256':digest(binary),'argv':list(map(str,compiler))};L.event('observation',dict(title='Heldout independentCPUreference build',motivation='Use unchanged U64all-document dot-product and canonical selection oracle',build=build,artifacts=[L.blob(work/'logs/compiler-version.stdout'),L.blob(work/'logs/reference-build.stdout'),L.blob(work/'logs/reference-build.stderr')]),[inputs['method_freeze']])
  oracle=Path(method['oracle_path']);assert str(oracle)==str(work/'full-top10.tsv');cfg=frozen['payload']['expected'];assert cfg==dict(mode='cpu-sequential',docs=8841823,query_count=6480,k=10,gpu_id=-1,index_path=inputs['gsp']['path'],queries_path=info['queries']['path'])
  spec=L.plan(dict(title='Frozen heldout independentU64oracle',question=inputs['calibration_question'],hypothesis='Complete canonical exact U64top10for every heldout query',motivation='Independent full-collection reference before all five-system correctness/timing',owner='vaibhav',expected=cfg,behavior=frozen['payload']['behavior'],split='heldout',freeze_ref=inputs['method_freeze'],inputs=[dict(role='canonical_gsp',**{k:inputs['gsp'][k] for k in ['path','sha256']}),dict(role='queries',**info['queries']),dict(role='query_split',**inputs['query_split'])],build=build,timing_boundary='Correctness-only independentCPUreference; no performance comparison',correctness_rule='All8841823documents eligible; score-desc originalnumericID-asc,fullU64products/sums',success_rule='EXECUTED and complete6480x10ordered results',failure_rule='Stop,preserve,review;no automatic technical repair',next_if_null='Retain result without tuning',analysis_plan='Independent source-row score replay; reference for every subsequent output',refs=[inputs['method_freeze']]))
  wait('before-independent-reference');state('independent-reference-pending')
  admission=work/'oracle-admission.jsonl';command=[sys.executable,root/'scripts/research.py','run',spec['id'],'--gate-seconds',900,'--output',oracle,'--output',admission,'--command',*controller.admission_command(root,admission,[binary,'score',inputs['gsp']['path'],info['queries']['path'],oracle,10,6480])];child('reference-run',command,86400)
  event=read(work/'logs/reference-run.stdout');assert event['payload']['status']=='EXECUTED' and not event['payload']['runtime_delta'];gold={};qids=[l.split()[0] for l in Path(info['queries']['path']).read_text().splitlines()]
  for row in csv.DictReader(oracle.open(),delimiter='\t'):
   values=gold.setdefault(row['query_id'],[]);assert int(row['rank'])==len(values)+1;values.append((int(row['original_id']),int(row['score'])))
  assert list(gold)==qids and all(len(v)==10 and len({d for d,s in v})==10 and v==sorted(v,key=lambda x:(-x[1],x[0])) for v in gold.values())
  # Independently recompute all64800 returned scores from originalGSProws.
  import numpy as np,struct
  with Path(inputs['gsp']['path']).open('rb') as f:assert f.read(8)==b'GSPIDX01';n,v,nnz=struct.unpack('<IIQ',f.read(16))
  ids=np.memmap(inputs['gsp']['path'],dtype='<u8',mode='r',offset=24,shape=n);offsets=np.memmap(inputs['gsp']['path'],dtype='<u8',mode='r',offset=24+8*n,shape=n+1);start=24+8*n+8*(n+1);terms=np.memmap(inputs['gsp']['path'],dtype='<u4',mode='r',offset=start,shape=nnz);impacts=np.memmap(inputs['gsp']['path'],dtype='u1',mode='r',offset=start+4*nnz,shape=nnz);order=np.argsort(ids);sorted_ids=ids[order];checked=0
  for line in Path(info['queries']['path']).read_text().splitlines():
   fields=line.split();q={int(t.split(':')[0]):int(t.split(':')[1]) for t in fields[1:]}
   for identifier,score in gold[fields[0]]:
    at=int(np.searchsorted(sorted_ids,np.uint64(identifier)));assert at<n and int(sorted_ids[at])==identifier;d=int(order[at]);lo,hi=int(offsets[d]),int(offsets[d+1]);assert score==sum(q.get(int(t),0)*int(w) for t,w in zip(terms[lo:hi],impacts[lo:hi]));checked+=1
  assert checked==64800
  review(L,event['id'],dict(result_summary=dict(ordered_query_groups=6480,source_row_scores_checked=checked),interpretation='Independent U64reference completed and returned-row scores replayed',conclusions=[dict(text='Independent U64reference produced complete canonical results and all64800returned-row scores match originalGSPdotproducts',scope='6480heldout queries,8841823passages,k10;correctness-only,no timing comparison',evidence_refs=[event['id']])],decision='Proceed with frozen five-system correctness and performance',motivation='Require complete reference before comparison',confidence='high',limitations=['Returned-row replay does not independently prove top-k; full all-document source plus five-system correctness retained'],next_steps=[dict(task='Complete all frozen heldout system runs',owner='unassigned',reason='Generalization evaluation',done_when='All correctness/timing outputs independently validated')]))
  del ids,offsets,terms,impacts,order,sorted_ids
  inputs.update(queries=info['queries'],oracle={'path':str(oracle),'sha256':digest(oracle)},prepared_queries_sha256=info['queries']['sha256'],prepared_oracle_sha256=digest(oracle),prepared_pisa_sha256=info['pisa']['sha256'],calibration_queries_sha256=CAL_SHA,evaluation_rule=method)
  resolved=root/'support/heldout-resolved-inputs.json';write_new(resolved,inputs);state('starting-five-system-heldout-controller')
  child('five-system-heldout',[sys.executable,root/'experiments/2026-11-18-matched-final-evaluation/code/run_heldout.py','--repository',root,'--inputs',resolved,'--output',out],259200)
  final=read(out/'status.json');assert final['state']=='heldout-complete-independent-analysis-pending';L.verify();export(L);state('complete-independent-analysis-pending',evaluation=final)
 except BaseException as error:
  L.event('observation',dict(title='Frozen heldout pipeline stopped',motivation='Preserve failure; do not tune or automatically resubmit',error=repr(error),heldout_exposed=exposed),[inputs['method_freeze']]);state('failed',error=repr(error));traceback.print_exc();raise
if __name__=='__main__':main()
