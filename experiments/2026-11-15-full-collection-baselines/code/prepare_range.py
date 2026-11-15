"""CPU-only full range-index construction; no query retrieval or GPU timing."""
import argparse,datetime,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
from prepare_inputs import digest
INDEX_SHA='e838dfbf78855ea48242f89ee9a8e3596830926e31f0864377c279f77fd754ac'

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rbp',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    repo=a.repository.resolve();out=a.output.resolve();rbp=a.rbp.resolve()
    if rbp.exists():raise ValueError('Existing index: inspect/reuse, do not duplicate')
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'build').mkdir();rbp.parent.mkdir(parents=True,exist_ok=True)
    manifest={'source_commit':a.source_commit,'commands':[],'memory_cap_bytes':32*1024**3,'qualification':'Uninstrumented CPU index build/verification; no retrieval or GPU timing.'}
    def status(stage,state='running',**kw):
        tmp=out/'status.pending';tmp.write_text(json.dumps(dict(stage=stage,state=state,pid=os.getpid(),updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));tmp.replace(out/'status.json')
    def run(tag,argv,cap=False):
        before=time.monotonic();manifest['commands'].append({'tag':tag,'argv':list(map(str,argv))})
        def limit():resource.setrlimit(resource.RLIMIT_AS,(manifest['memory_cap_bytes'],manifest['memory_cap_bytes']))
        with (out/'logs'/(tag+'.stdout.log')).open('x') as stdout,(out/'logs'/(tag+'.stderr.log')).open('x') as stderr:
            result=subprocess.run(list(map(str,argv)),cwd=repo,stdout=stdout,stderr=stderr,preexec_fn=limit if cap else None)
        manifest['commands'][-1].update(returncode=result.returncode,elapsed_seconds=time.monotonic()-before)
        if result.returncode:raise RuntimeError(tag+' failed; inspect preserved logs')
    status('verify_input_source')
    try:
        receipt=json.loads((repo/'deployment-receipt.json').read_text())
        if receipt['source_commit']!=a.source_commit:raise ValueError('commit mismatch')
        for name,sha in receipt['files'].items():
            if not (repo/name).resolve().is_relative_to(repo) or digest(repo/name)!=sha:raise ValueError('source hash mismatch')
        if digest(a.index)!=INDEX_SHA:raise ValueError('full GSP hash mismatch')
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
        if available<36*1024**3:raise RuntimeError('insufficient current host headroom for32GiB capped range construction')
        if shutil.disk_usage(rbp.parent).free<16*1024**3:raise RuntimeError('insufficient scratch')
        manifest['host_available_before_build']=available;manifest['input_sha256']=INDEX_SHA
        compiler=shutil.which('g++');tool=out/'build/range_bitpack_tool'
        if compiler is None:raise RuntimeError('g++ unavailable')
        status('build_tool');run('compiler-version',[compiler,'--version']);run('build',[compiler,'-O3','-std=c++17','-pthread',repo/'src/range_bitpack_tool.cpp','-o',tool]);manifest['binary_sha256']=digest(tool)
        status('independent_cpu_fixture');run('fixture',[sys.executable,Path(__file__).with_name('check_range_inputs.py'),tool,repo])
        status('full_build_and_roundtrip');run('full-build',['/usr/bin/time','-v',tool,'build',a.index,rbp,'1024','positions'],cap=True)
        status('reload_and_every_posting_verify');run('full-verify',['/usr/bin/time','-v',tool,'verify',a.index,rbp],cap=True)
        status('hash_output');manifest['output']={'path':str(rbp),'sha256':digest(rbp),'bytes':rbp.stat().st_size};manifest['source_files']=receipt['files']
        if digest(a.index)!=INDEX_SHA:raise ValueError('GSP changed during construction')
        with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
        status('complete','complete',output=manifest['output'])
    except Exception as error:
        manifest['error']={'type':type(error).__name__,'message':str(error)}
        with (out/'failure-manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
        status('failed','failed',error=manifest['error']);raise

if __name__=='__main__':main()
