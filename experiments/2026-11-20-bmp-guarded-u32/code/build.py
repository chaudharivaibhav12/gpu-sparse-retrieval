"""CPU-only build/unit-test preparation; never reads retrieval inputs or starts CUDA."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from materialize import materialize,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    assert len(a.source_commit)==40 and all(c in '0123456789abcdef' for c in a.source_commit)
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);logs=root/'logs';logs.mkdir();manifest={'status':'building','source_commit':a.source_commit,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':[],'scope':'CPU compilation/unit checks only;no dataset/index/query open,no retrieval measurement,no GPU execution'}
    env=dict(os.environ,RUSTFLAGS='',CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1');env.pop('CARGO_ENCODED_RUSTFLAGS',None)
    def run(argv,tag,cwd,timeout=1800):
        row=dict(argv=argv,cwd=str(cwd),status='started',environment_overrides={k:env[k] for k in ['RUSTFLAGS','CARGO_BUILD_JOBS','RAYON_NUM_THREADS','OMP_NUM_THREADS']});manifest['commands'].append(row)
        try:
            with (logs/(tag+'.stdout')).open('xb') as so,(logs/(tag+'.stderr')).open('xb') as se:r=subprocess.run(argv,cwd=cwd,env=env,stdout=so,stderr=se,timeout=timeout)
            row.update(status='finished',returncode=r.returncode);assert r.returncode==0,(tag,r.returncode)
        except BaseException as error:row.update(status='failed',error=repr(error));raise
        finally:
            for stream in ['stdout','stderr']:
                file=logs/(tag+'.'+stream)
                if file.exists():row[stream+'_sha256']=sha(file)
    try:
        manifest['materialization']=materialize(a.parent,root/'BMP',Path(__file__).resolve().parents[1]);run(['rustc','-vV'],'rustc',root);run(['cargo','--version'],'cargo',root)
        run(['cargo','test','--locked','--offline','--lib','--bin','project_benchmark_u32','--bin','project_benchmark_u64'],'unit-tests',root/'BMP')
        run(['cargo','build','--release','--locked','--offline','--bin','project_benchmark_u32','--bin','project_benchmark_u64'],'release-build',root/'BMP')
        generated=root/'generated-proto';generated.mkdir();manifest['generated_sources']=[]
        for file in sorted((root/'BMP/target').glob('*/build/bmp-*/out/common_index_format_v1.rs')):
            h=sha(file);copy=generated/(h+'.rs')
            if not copy.exists():copy.write_bytes(file.read_bytes())
            manifest['generated_sources'].append({'original_path':str(file),'path':str(copy),'sha256':h})
        assert len(manifest['generated_sources'])>=2,'Both test/release protobuf codegen must be captured'
        for bits in [32,64]:run([str(root/'BMP/target/release'/('project_benchmark_u'+str(bits))),'--help'],'help-u'+str(bits),root)
        manifest['binaries']={str(bits):{'path':str(root/'BMP/target/release'/('project_benchmark_u'+str(bits))),'sha256':sha(root/'BMP/target/release'/('project_benchmark_u'+str(bits)))} for bits in [32,64]};manifest['status']='BUILD_AND_UNIT_TESTS_PASS_NO_MEASUREMENTS'
    except BaseException as e:manifest['status']='BUILD_PREPARATION_FAILED';manifest['error']=repr(e);raise
    finally:
        manifest['logs']=[{'path':str(f),'sha256':sha(f)} for f in sorted(logs.iterdir())];(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':manifest['status'],'source_commit':a.source_commit,'commands':len(manifest['commands'])}),flush=True)
if __name__=='__main__':main()
