"""CPU-only regression against the successful compact parent's actual runtime."""
import importlib.util,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
B=Path(__file__).resolve().parents[1]
code=ROOT/'experiments/2026-11-17-gpusparse-u8-storage/code'
sys.path.insert(0,str(code))
spec=importlib.util.spec_from_file_location('compact_expected',code/'fp32/expected_config.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
parent=json.loads((B/'configs/compact-parent-runtime.json').read_text())['runtime']['config']
profile=json.loads((B/'configs/device-profile.json').read_text());assert profile['gpu_name']==parent['gpu_name'] and profile['compute_capability']==parent['compute_capability']
versions=['torch_version','triton_version','numpy_version','scipy_version','torch_cuda_version']
software={k:parent[k] for k in versions};complete=dict(software,gpu_name=profile['gpu_name'],compute_capability=profile['compute_capability'])
with tempfile.TemporaryDirectory() as directory:
    # Header-only parser fixture: never described as a complete GSP or a run.
    index=Path(directory)/'header-only.gsp';index.write_bytes(b'GSPIDX01'+struct.pack('<IIQ',parent['docs'],parent['vocab'],parent['nnz']))
    def resolve(environment,repeats,warmups,verify):
        return module.expected(index,ROOT/'results/calibration-500.tsv',parent['reference_path'],parent['native_repo'],code/'fp32/canonical_baseline.py',environment,0,500,10,512,repeats,warmups,verify,{'triton_kernel.py':parent['native_monolithic_sha256']},parent['native_revision'])
    try:resolve(software,1,0,0)
    except KeyError as e:assert e.args==('gpu_name',)
    else:raise AssertionError('Missing hardware metadata was silently defaulted')
    actual=resolve(complete,parent['repeats'],parent['warmup_rounds'],parent['verify_scores'])
    paths={'index_path','queries_path','reference_path','native_repo'}
    assert {k:v for k,v in actual.items() if k not in paths}=={k:v for k,v in parent.items() if k not in paths}
    for repeats,warmups,verify in [(1,0,1),(20,3,0)]:
        config=resolve(complete,repeats,warmups,verify)
        assert config['gpu_name']==profile['gpu_name'] and config['compute_capability']==profile['compute_capability']
        assert (config['repeats'],config['warmup_rounds'],config['verify_scores'],config['requested_k'],config['query_count'])==(repeats,warmups,verify,10,500)
print(json.dumps({'status':'PASS','checks':4,'scope':'missing-key regression, complete actual-parent snapshot roundtrip and both current run modes; CPU-only metadata resolution'}))
