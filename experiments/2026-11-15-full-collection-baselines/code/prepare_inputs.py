"""Build and validate the unchanged importer, then prepare one full GSP.

Uninstrumented build/import observation, not a retrieval benchmark. Logs and
source/input/output hashes are retained. No prepared query is read or changed.
"""
import argparse,datetime,hashlib,json,os,resource,shutil,struct,subprocess,sys,time
from pathlib import Path
PINNED_SHA='16b111a409c557b966322c3b093e96a83c54a76e7863a0af4db3d0ae57043953'
PINNED_BYTES=6575573728
N=8841823
V=28679

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,required=True);parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gsp',type=Path,required=True);parser.add_argument('--source-commit',required=True);parser.add_argument('--memory-cap-gib',type=int,default=36)
    a=parser.parse_args();repo=a.repository.resolve();out=a.output.resolve();gsp=a.gsp.resolve()
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('exact source commit required')
    if not 1<=a.memory_cap_gib<=40:raise ValueError('unsafe host memory cap')
    if gsp.exists():raise ValueError('existing GSP: reuse/inspect before another import')
    if a.raw.resolve()==gsp:raise ValueError('output aliases raw input')
    gsp.parent.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'build').mkdir()
    status=out/'status.json';manifest={'qualification':'Uninstrumented importer build/fixture/import observation; no retrieval or GPU timing.','source_commit':a.source_commit,'started_at_utc':now(),'commands':[],'memory_cap_bytes':a.memory_cap_gib*1024**3}
    def save(stage,state='running',**kw):
        data=dict(stage=stage,state=state,pid=os.getpid(),updated_at_utc=now(),**kw);temp=out/'status.pending'
        with temp.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
        temp.replace(status)
    def run(tag,argv,cap=False):
        manifest['commands'].append({'tag':tag,'argv':list(map(str,argv)),'cwd':str(repo),'started_at_utc':now()})
        def limit():resource.setrlimit(resource.RLIMIT_AS,(manifest['memory_cap_bytes'],manifest['memory_cap_bytes']))
        with (out/'logs'/('%s.stdout.log'%tag)).open('x') as stdout,(out/'logs'/('%s.stderr.log'%tag)).open('x') as stderr:
            p=subprocess.run(list(map(str,argv)),cwd=repo,stdout=stdout,stderr=stderr,preexec_fn=limit if cap else None)
        manifest['commands'][-1].update(returncode=p.returncode,finished_at_utc=now())
        if p.returncode:raise RuntimeError('%s failed (%d); inspect saved logs'%(tag,p.returncode))
    save('verify_input_and_source')
    try:
        receipt=json.loads((repo/'deployment-receipt.json').read_text())
        if receipt['source_commit']!=a.source_commit:raise ValueError('deployment commit mismatch')
        required={'src/cpu_pilot.cpp','src/research_runtime.h','tests/check_cpu_pilot.py',str(Path(__file__).resolve().relative_to(repo))}
        if not required.issubset(receipt['files']):raise ValueError('incomplete deployment receipt')
        for relative,expected in receipt['files'].items():
            if not (repo/relative).resolve().is_relative_to(repo):raise ValueError('source receipt path escapes checkout')
            if digest(repo/relative)!=expected:raise ValueError('source hash mismatch: '+relative)
        manifest['deployment']=receipt
        if a.raw.stat().st_size!=PINNED_BYTES or digest(a.raw)!=PINNED_SHA:raise ValueError('raw CIFF mismatch')
        manifest['input']={'path':str(a.raw),'sha256':PINNED_SHA,'bytes':PINNED_BYTES}
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        manifest['host_available_before_build']=available
        if available<manifest['memory_cap_bytes']+4*1024**3:raise RuntimeError('insufficient current host headroom for capped import')
        if shutil.disk_usage(gsp.parent).free<16*1024**3:raise RuntimeError('insufficient scratch headroom')
        save('build');compiler=shutil.which('g++')
        if compiler is None:raise RuntimeError('g++ unavailable')
        run('compiler-version',[compiler,'--version'])
        binary=out/'build/cpu_pilot'
        run('build',[compiler,'-O3','-std=c++17','-pthread',repo/'src/cpu_pilot.cpp','-o',binary])
        manifest['binary_sha256']=digest(binary)
        save('independent_import_fixture');run('fixture',[sys.executable,Path(__file__).with_name('check_import.py'),binary,repo])
        save('full_import');timing=shutil.which('/usr/bin/time')
        if timing is None:raise RuntimeError('/usr/bin/time unavailable for import resource evidence')
        before=time.monotonic();run('full-import',[timing,'-v',binary,'import',a.raw,gsp,N],cap=True);manifest['import_elapsed_seconds']=time.monotonic()-before
        save('verify_output_hash')
        with gsp.open('rb') as f:
            if f.read(8)!=b'GSPIDX01':raise ValueError('bad output magic')
            n,v,nnz=struct.unpack('<IIQ',f.read(16))
        if n!=N or v!=V or nnz>1049282186:raise ValueError('output header incompatible with metadata')
        manifest['output']={'path':str(gsp),'docs':n,'vocab':v,'nnz':nnz,'bytes':gsp.stat().st_size,'sha256':digest(gsp)}
        if digest(a.raw)!=PINNED_SHA:raise ValueError('raw input changed during import')
        manifest['finished_at_utc']=now()
        with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
        save('complete','complete',output=manifest['output'])
    except Exception as error:
        manifest['error']={'type':type(error).__name__,'message':str(error)}
        with (out/'failure-manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
        save('failed','failed',error=manifest['error']);raise

if __name__=='__main__':main()
