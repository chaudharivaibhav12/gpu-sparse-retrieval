"""Validate the full index and create an instrumented CPU top-10 reference.

Requires the project's existing research runner/policy in the deployed source.
No CUDA retrieval, held-out queries or performance comparison is executed.
"""
import argparse,csv,datetime,json,os,platform,shutil,struct,subprocess,sys
from pathlib import Path
import numpy as np
from validate_index import validate

QUERY_SHA='150917d784f009fddb453a1635bc62d3ec238c4a101347d46914adaad371d091'
SPLIT_SHA='650a121bf25d71beee4e2be8d05228ce8d700b2a866d8c8ba996e0ec53ce33d1'

def make_fixture(root):
    ids=[2**64-1,7,0,3,20];rows=[{0:255,2:1},{0:255,2:1},{},{1:16},{2:2}]
    terms=[];weights=[];offsets=[0]
    for row in rows:
        for t,w in sorted(row.items()):terms.append(t);weights.append(w)
        offsets.append(len(terms))
    gsp=root/'fixture.gsp';query=root/'fixture.tsv';dictionary=b''.join(struct.pack('<I',2)+('t%d'%t).encode() for t in range(3))
    gsp.write_bytes(b'GSPIDX01'+struct.pack('<IIQ',len(ids),3,len(terms))+struct.pack('<5Q',*ids)+struct.pack('<6Q',*offsets)+struct.pack('<%dI'%len(terms),*terms)+bytes(weights)+dictionary)
    query.write_text('empty\ntie 0:32 2:1\n')
    expected={}
    for qid,q in [('empty',{}),('tie',{0:32,2:1})]:
        expected[qid]=sorted([(identifier,sum(q.get(t,0)*w for t,w in row.items())) for identifier,row in zip(ids,rows)],key=lambda x:(-x[1],x[0]))
    return gsp,query,expected

def read_tops(path,queries,k):
    groups={qid:[] for qid in queries}
    with Path(path).open() as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames!=['query_id','rank','original_id','score']:raise ValueError('oracle header')
        for row in reader:
            qid=row['query_id']
            if qid not in groups or int(row['rank'])!=len(groups[qid])+1:raise ValueError('query or rank sequence')
            groups[qid].append((int(row['original_id']),int(row['score'])))
    for values in groups.values():
        if len(values)!=k or len({d for d,s in values})!=k or any(s<0 or s>=2**64 or d<0 or d>=2**64 for d,s in values):raise ValueError('result count/domain')
        if values!=sorted(values,key=lambda x:(-x[1],x[0])):raise ValueError('canonical ranking')
    return groups

def check_returned_scores(index,queries,tops):
    with Path(index).open('rb') as f:
        if f.read(8)!=b'GSPIDX01':raise ValueError('GSP magic')
        n,v,nnz=struct.unpack('<IIQ',f.read(16))
    ids=np.memmap(index,dtype='<u8',mode='r',offset=24,shape=n);offsets=np.memmap(index,dtype='<u8',mode='r',offset=24+8*n,shape=n+1)
    start=24+8*n+8*(n+1);terms=np.memmap(index,dtype='<u4',mode='r',offset=start,shape=nnz);weights=np.memmap(index,dtype='u1',mode='r',offset=start+4*nnz,shape=nnz)
    order=np.argsort(ids);sorted_ids=ids[order];checked=0
    for qid,values in tops.items():
        q=queries[qid]
        for identifier,score in values:
            where=int(np.searchsorted(sorted_ids,np.uint64(identifier)))
            if where>=n or int(sorted_ids[where])!=identifier:raise ValueError('returned ID outside corpus')
            d=int(order[where]);lo,hi=int(offsets[d]),int(offsets[d+1])
            actual=sum(q.get(int(t),0)*int(w) for t,w in zip(terms[lo:hi],weights[lo:hi]))
            if actual!=score:raise ValueError('independent returned-score mismatch')
            checked+=1
    return checked

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    repo=a.repository.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'build').mkdir()
    os.chdir(repo);sys.path.insert(0,str(repo/'scripts'));from research import Ledger,digest,read,write_new,review
    ledger=Ledger(repo);status=out/'status.json';question=None;run=None
    def state(stage,kind='running',**extra):
        temp=out/'status.pending';temp.write_text(json.dumps(dict(state=kind,stage=stage,pid=os.getpid(),source_commit=a.source_commit,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2));temp.replace(status)
    def child(tag,argv):
        with (out/'logs'/(tag+'.stdout.log')).open('x') as stdout,(out/'logs'/(tag+'.stderr.log')).open('x') as stderr:
            result=subprocess.run(list(map(str,argv)),stdout=stdout,stderr=stderr)
        if result.returncode:raise RuntimeError(tag+' failed; inspect saved logs')
    state('verify_source_inputs')
    try:
        receipt=read(repo/'deployment-receipt.json')
        if receipt['source_commit']!=a.source_commit:raise ValueError('source commit mismatch')
        for relative,sha in {**receipt['files'],**receipt.get('support_files',{})}.items():
            if not (repo/relative).resolve().is_relative_to(repo) or digest(repo/relative)!=sha:raise ValueError('source/support hash mismatch')
        if digest(a.queries)!=QUERY_SHA:raise ValueError('calibration query hash')
        split_sha=digest(a.split);metadata=read(a.metadata)
        if split_sha!=SPLIT_SHA:raise ValueError('calibration/held-out split identity')
        if digest(a.raw)!=metadata['source_receipt']['sha256']:raise ValueError('raw CIFF hash')
        index_sha=digest(a.index)
        questions={'title':'Full-collection exact baseline qualification','motivation':'Verify8841823passages and establish CPUreference before matched GPU comparisons.','parent_experiment':'2026-11-15-range-worker-sweep','parent_merge':'201be6998bace088900e1b7ccb961bb5277273e4','source_commit':a.source_commit,'qualification':'New thin execution ledger; public parent snapshot/merge identity is preserved without claiming inherited instrumentation.','artifacts':[ledger.blob(Path(__file__)),ledger.blob(Path(__file__).with_name('validate_index.py')),ledger.blob(a.metadata),ledger.blob(repo/'deployment-receipt.json')]}
        question=ledger.event('question',questions)
        state('validator_fixture');child('validator-fixture',[sys.executable,Path(__file__).with_name('check_index_validator.py'),repo])
        state('independent_full_index_validation');validation=validate(a.index,a.raw,metadata);write_new(out/'index-validation.json',validation)
        observation=ledger.event('observation',dict(title='Independent full-index validation',motivation='Check full input before reference retrieval.',result=validation,artifacts=[ledger.blob(out/'index-validation.json')]),[question['id']])
        state('build_cpu_reference');compiler=shutil.which('g++');binary=out/'build/cpu_pilot'
        if compiler is None:raise RuntimeError('g++ missing')
        argv=[compiler,'-O3','-std=c++17','-pthread',repo/'src/cpu_pilot.cpp','-o',binary];child('compiler-version',[compiler,'--version']);child('build',argv)
        build=dict(source_commit=a.source_commit,source_sha256=digest(repo/'src/cpu_pilot.cpp'),runtime_sha256=digest(repo/'src/research_runtime.h'),binary_sha256=digest(binary),argv=list(map(str,argv)),compiler=(out/'logs/compiler-version.stdout.log').read_text(),python=sys.version,platform=platform.platform())
        build_event=ledger.event('observation',dict(title='CPUreference build',motivation='Build exact committed source before instrumented correctness retrieval.',build=build,artifacts=[ledger.blob(out/'logs/build.stdout.log'),ledger.blob(out/'logs/build.stderr.log'),ledger.blob(out/'logs/compiler-version.stdout.log')]),[question['id']])
        policy=read(repo/'research/policy.json')['sticky_behavior']
        def one(tag,index,query,split,k,count,expected_fixture=None):
            nonlocal run
            target=out/(tag+'-top10.tsv');expected=dict(mode='cpu-sequential',docs=5 if split=='fixture' else metadata['docs'],query_count=count,k=k,gpu_id=-1,index_path=str(index),queries_path=str(query))
            inputs=[dict(role=role,path=str(path),sha256=digest(path)) for role,path in [('index',index),('queries',query)]]
            if split=='calibration':inputs.append(dict(role='query_split',path=str(a.split),sha256=split_sha))
            spec=ledger.plan(dict(title='Full baseline CPU '+tag,question=question['id'],hypothesis='Unchanged exhaustive U64 scoring preserves exact scores, all-document eligibility and numeric-ID ties.',motivation='Supply independent exact reference before GPU performance comparisons.',owner='vaibhav',expected=expected,behavior=policy,success_rule='Executed accepted runtime; complete canonical outputs; independent fixture/full returned-score validation.',failure_rule='Any gate/output/domain/score mismatch blocks baseline timing.',next_if_null='Preserve failure and inspect before reviewed repair.',split=split,timing_boundary='Correctness reference only; no latency comparison.',correctness_rule='Hand-computed fixture before full reference; all returned real scores independently recomputed; GPU comparison later.',analysis_plan='No performance claim. Full top-k optimality relies on exhaustive CPU implementation, fixtures and later independent GPU comparison.',build=build,inputs=inputs,parent_external='2026-11-15-range-worker-sweep',declared_changes={'corpus':'Full corpus instead of 1M; exact IDs/weights/scoring/ties unchanged.'},refs=[question['id'],build_event['id'],observation['id']]))
            state('cpu_'+tag)
            command=[sys.executable,str(repo/'scripts/research.py'),'run',spec['id'],'--gate-seconds','600','--output',str(target),'--command',str(binary),'score',str(index),str(query),str(target),str(k),str(count)]
            child(tag+'-runner',command)
            finished=json.loads((out/'logs'/(tag+'-runner.stdout.log')).read_text());run=finished['id']
            if finished['payload']['status']!='EXECUTED':raise ValueError('reference runtime not executed')
            actual=read_tops(target,expected_fixture or queries,min(k,expected['docs']))
            if expected_fixture is not None:
                if actual!=expected_fixture:raise ValueError('hand-computed fixture mismatch')
                checked=sum(map(len,actual.values()))
            else:checked=check_returned_scores(index,queries,actual)
            review(ledger,run,dict(result_summary={'output_groups':len(actual),'independent_checked_scores':checked},interpretation='Reference qualification passed on declared input; no performance assertion.',conclusions=[dict(text='Exact returned scores and canonical ordering passed declared checks.',scope=split+' correctness reference only; '+str(expected['docs'])+' passages',evidence_refs=[run])],decision='Continue qualification',motivation='Preserve validation and its scope.',confidence='high',limitations=['Full source posting bodies not independently replayed.','Full top-k optimality is not proven by checking only returned scores; exhaustive implementation and fixtures supply that foundation.'],next_steps=[dict(task='Match GPU outputs before timing.',owner='unassigned',reason='Independent implementation check.',done_when='Zero canonical mismatches.')],refs=[question['id']]))
            return target,run
        fixture=out/'fixture';fixture.mkdir();fg,fq,expected=make_fixture(fixture);queries={}
        one('fixture',fg,fq,'fixture',10,2,expected)
        with a.queries.open() as f:
            for line in f:
                fields=line.split();qid=fields[0]
                if qid in queries:raise ValueError('duplicate query ID')
                items=[tuple(map(int,x.split(':'))) for x in fields[1:]]
                if len({t for t,w in items})!=len(items) or any(t<0 or t>=metadata['vocab'] or not 0<=w<=32 for t,w in items):raise ValueError('query domain')
                queries[qid]=dict(items)
        if len(queries)!=500:raise ValueError('calibration membership/count')
        target,reference_run=one('full',a.index,a.queries,'calibration',10,500)
        if digest(a.index)!=index_sha or digest(a.queries)!=QUERY_SHA:raise ValueError('input changed')
        result={'source_commit':a.source_commit,'index_sha256':index_sha,'oracle_path':str(target),'oracle_sha256':digest(target),'reference_run':reference_run,'question':question['id'],'query_count':500,'docs':metadata['docs'],'qualification':'Instrumented CPU correctness reference; no GPU speedup, held-out or source-posting replay claim.'}
        write_new(out/'manifest.json',result);write_new(out/'ledger-verify.json',ledger.verify());state('complete','complete',result=result)
    except Exception as error:
        events=ledger.events();reviewed={e['payload'].get('run') for e in events if e['kind']=='review'}
        pending=[e['id'] for e in events if e['kind']=='run_finished' and e['id'] not in reviewed]
        ledger.event('observation',dict(title='Full-reference preparation failure',motivation='Preserve technical failure before any repair.',error={'type':type(error).__name__,'message':str(error)},runs_pending_review=pending,qualification='No full baseline performance claim.'),[question['id']] if question else [])
        if pending:ledger.event('task',dict(title='Review stopped reference attempts',motivation='Every stopped/failed attempt must be inspected before repair.',runs=pending,next_action='Inspect raw logs/config/results and append scoped reviews before another submission.'),pending)
        state('failed','failed',error={'type':type(error).__name__,'message':str(error)});raise

if __name__=='__main__':main()
