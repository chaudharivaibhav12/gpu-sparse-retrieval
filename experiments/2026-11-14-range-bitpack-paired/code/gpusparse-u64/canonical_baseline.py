"""Contract-aligned adapted GPUSparse monolithic scorer; U64 arithmetic and numeric-ID ties."""
import argparse,csv,importlib,inspect,json,math,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from gsp_input import load_index,read_queries,result_score,padded_query,sha256
from runtime_gate import runtime_report,write_new
REV='92bb87b36f10e78ab39c81e9db2412dbb7a3b650'
HASHES={'triton_kernel.py':'ef298aba741b6fe86a31021369b808549fa465c7011444cb98770808bcf79fd0'}
CONTRACT={'contract':'GSP-CAP32-U64-NUMERIC-TIES-v1','arithmetic':'uint64','document_impacts':'uint8-exact','query_policy':'BMP-f32-ceil-cap32','tie_order':'score_desc_original_numeric_id_asc','eligibility':'all_documents'}

def references(path,k):
 path=Path(path)
 if path.suffix=='.json':return {q:[(int(x[0]),int(x[1])) for x in values[:k]] for q,values in json.loads(path.read_text(encoding='utf-8')).items()}
 result={}
 with path.open(encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f,delimiter='\t'):result.setdefault(row['query_id'],[]).append((int(row['original_id']),int(row['score'])))
 return {q:v[:k] for q,v in result.items()}

def arguments():
 p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['monolithic'])
 for name in ['index','queries','reference','native-repo','csv','top','validation','jit-manifest']:p.add_argument('--'+name,required=True)
 for name in ['limit','k','gpu','block-pl','repeats','warmups','verify']:p.add_argument('--'+name,type=int,required=True)
 a=p.parse_args()
 if min(a.gpu,a.warmups)<0 or a.limit<1 or not 1<=a.k<=128 or a.repeats<1 or a.block_pl<1 or a.block_pl&(a.block_pl-1) or a.verify not in [0,1]:p.error('Unsupported explicit settings; query limit and requested k must be positive for the native audit')
 return a

def main(a):
 import numpy as np,scipy,torch,triton
 native=Path(a.native_repo).resolve()
 assert subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD'],text=True).strip()==REV
 subprocess.run(['git','-C',str(native),'diff','--quiet','HEAD','--','src'],check=True)
 for name,value in HASHES.items():assert sha256(native/'src'/name)==value,name
 # Original repository is provenance only; the labelled adapted module implements this treatment.
 import u64_monolithic as mono
 assert (torch.__version__,triton.__version__,np.__version__,scipy.__version__)==('2.6.0+cu126','3.2.0','2.2.6','1.15.3')
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and a.gpu==0
 cache=Path(os.environ['TRITON_CACHE_DIR']).resolve();cache.mkdir(parents=True,exist_ok=True)
 cache_before={str(p):sha256(p) for p in cache.rglob('*') if p.is_file()}
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(a.gpu)
 index=load_index(a.index);queries=read_queries(a.queries,index['V'],a.limit);wanted=min(a.k,index['N']);reference=references(a.reference,wanted)
 assert queries and all(q in reference and len(reference[q])==wanted for q,_ in queries)
 width=max(1,max((len(v) for _,v in queries),default=0));csc=index['csc'];device=torch.device('cuda',a.gpu)
 gpu=SimpleNamespace(device=device,num_docs=index['N'],vocab_size=index['V'],doc_ids=torch.from_numpy(csc.indices.astype(np.int32)).to(device),scores=torch.from_numpy(csc.data.astype(np.float32)).to(device),offsets=torch.from_numpy(csc.indptr[:-1].astype(np.int64)).to(device),lengths=torch.from_numpy(np.diff(csc.indptr).astype(np.int32)).to(device))
 order=np.argsort(index['ids'],kind='stable');ranks=np.empty(index['N'],dtype=np.int32);ranks[order]=np.arange(index['N'],dtype=np.int32)
 gpu.id_ranks=torch.from_numpy(ranks).to(device)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 cfg=dict(mode='gpusparse-u64-canonical',variant='monolithic',docs=index['N'],vocab=index['V'],nnz=index['nnz'],query_count=len(queries),max_query_terms=max((len(v) for _,v in queries),default=0),query_tensor_width=width,requested_k=a.k,returned_depth=wanted,batch_size=1,max_supported_k=128,gpu_id=a.gpu,gpu_name=torch.cuda.get_device_name(a.gpu),compute_capability='.'.join(map(str,torch.cuda.get_device_capability(a.gpu))),block_pl=a.block_pl,index_gpu_bytes=sum(x.numel()*x.element_size() for x in [gpu.doc_ids,gpu.scores,gpu.offsets,gpu.lengths,gpu.id_ranks]),canonical_rank_bytes=gpu.id_ranks.numel()*gpu.id_ranks.element_size(),dense_score_bytes=index['N']*8,packed_key_bytes=index['N']*8,index_path=a.index,queries_path=a.queries,reference_path=a.reference,native_repo=str(native),repeats=a.repeats,warmup_rounds=a.warmups,verify_scores=a.verify,max_conservative_score_bound=max((255*sum(w for _,w in t) for _,t in queries),default=0),selection_score_limit=2**31-1,selection_score_bits=mono.SCORE_BITS,selection_id_bits=mono.ID_BITS,pack_threads=mono.PACK_THREADS,native_revision=REV,native_monolithic_sha256=HASHES['triton_kernel.py'],adapted_kernel_sha256=sha256(Path(mono.__file__)),adapter_sha256=sha256(__file__),input_adapter_sha256=sha256(Path(__file__).with_name('gsp_input.py')),runtime_gate_sha256=sha256(Path(__file__).with_name('runtime_gate.py')),torch_version=str(torch.__version__),triton_version=triton.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__,torch_cuda_version=torch.version.cuda,document_gpu_dtype=str(gpu.scores.dtype),posting_gpu_dtype=str(gpu.doc_ids.dtype),offset_gpu_dtype=str(gpu.offsets.dtype),length_gpu_dtype=str(gpu.lengths.dtype),rank_gpu_dtype=str(gpu.id_ranks.dtype),accumulator_dtype=str(mono.ACCUMULATOR_DTYPE),selection_key_dtype=str(mono.KEY_DTYPE),score_output_dtype='torch.int64',cpu_affinity=','.join(map(str,sorted(os.sched_getaffinity(0)))),torch_num_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),grad_enabled=torch.is_grad_enabled())
 behavior=dict(CONTRACT,scoring='adapted-gpusparse-monolithic-u64',selection='unique-packed-int64-key-torch-topk',canonical_adaptation='u64-products-and-sums;original-uint64-IDrank;31scorebits+32tie_bits',query_terms='all-retained-global-padding-minus1',posting_order='ascending-internal-document-position',integer_domain_guard='uint8-impacts;unique-cap32-terms;255*sumweights<=2^31-1',query_validation='all-file-rows-before-limit',cpu_validation='perquery-plus-post' if a.verify else 'post-timing-all-outputs',timing='query-padding-and-upload-through-adapted-call-pack-select-decode-resultcopy-and-originalIDmap',index_setup='excluded;CSR-toCSC+originalIDrank+GPU-upload',native_source='derived-from-pinned-monolithic;u64-products-accumulator+canonical-packed-selector',requested_k_domain='positive-request-through128;empty-index-returns-zero-depth',key_order='score-desc-original-numeric-ID-asc;unique-positive-int64;no-key-overflow')
 runtime_report(cfg,behavior)
 paths=[Path(x) for x in [a.csv,a.top,a.validation,a.jit_manifest]]
 for p in paths:assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 fields=['phase','round','query_id','variant','docs','k','query_wall_ms','gpu_span_ms','result_count','position_count','invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors','canonical_id_mismatches','true_score_errors']
 retained=[];checks=[];true_cache={};begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 def emit_jit():
  after={str(p):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in cache.rglob('*') if p.is_file()}
  write_new(paths[3],dict(cache=str(cache),files=after,changed=[p for p,v in after.items() if cache_before.get(p)!=v['sha256']],qualification='Actual Triton cache/JIT artifact hashes after approved native compute; original artifacts must be preserved by the run driver. Torch CUDA binary provenance is pinned by environment/wheel hashes.'))
 def true_errors(qi,positions,values):
  bad=0
  for pos,value in zip(positions,values):
   if not 0<=pos<index['N']:bad+=1;continue
   key=(qi,pos)
   if key not in true_cache:true_cache[key]=result_score(index,pos,queries[qi][1])
   bad+=value!=true_cache[key]
  return bad
 with paths[0].open('x',encoding='utf-8',newline='') as c,paths[1].open('x',encoding='utf-8',newline='') as t:
  writer=csv.DictWriter(c,fieldnames=fields,lineterminator='\n');writer.writeheader();tops=csv.writer(t,delimiter='\t',lineterminator='\n');tops.writerow(['phase','round','variant','query_id','rank','original_id','score','internal_doc_position'])
  for phase,rounds in [('warmup',a.warmups),('measured',a.repeats)]:
   for repeat in range(rounds):
    for qi,(qid,terms) in enumerate(queries):
     wall=time.perf_counter_ns();host_ids,host_weights=padded_query(terms,width);begin.record()
     ti=torch.from_numpy(host_ids).to(device);tw=torch.from_numpy(host_weights).to(device)
     values,positions=mono.canonical_monolithic(gpu,ti,tw,top_k=a.k,block_pl=a.block_pl)
     end.record();scores=values.cpu().tolist()[0];docs=positions.cpu().tolist()[0]
     original=[int(index['ids'][d]) if 0<=d<index['N'] else -1 for d in docs]
     torch.cuda.synchronize();wall_ms=(time.perf_counter_ns()-wall)/1e6;span=begin.elapsed_time(end)
     target=reference[qid];invalid=sum(not 0<=d<index['N'] for d in docs);duplicate=len(docs)-len(set(docs));noninteger=sum(not isinstance(v,int) or v<0 for v in scores)
     sequence=sum(v!=expected[1] for v,expected in zip(scores,target))+abs(len(scores)-len(target));canonical=sum(d!=expected[0] for d,expected in zip(original,target))+abs(len(original)-len(target))
     per_true=true_errors(qi,docs,scores) if a.verify else 'post_validation'
     row=dict(phase=phase,round=repeat,query_id=qid,variant=a.variant,docs=index['N'],k=a.k,query_wall_ms=wall_ms,gpu_span_ms=span,result_count=len(scores),position_count=len(docs),invalid_positions=invalid,duplicate_results=duplicate,noninteger_scores=noninteger,score_sequence_errors=sequence,canonical_id_mismatches=canonical,true_score_errors=per_true)
     writer.writerow(row)
     for rank,(d,v,pos) in enumerate(zip(original,scores,docs),1):tops.writerow([phase,repeat,a.variant,qid,rank,d,repr(v),pos])
     retained.append((qi,docs,scores));checks.append(row)
     if canonical or invalid or duplicate or noninteger or sequence or len(scores)!=wanted or len(docs)!=wanted or (a.verify and per_true):
      c.flush();t.flush();write_new(paths[2],dict(success=False,query_id=qid,phase=phase,round=repeat,failed_row=row,qualification='Stopped immediately on observed canonical score/ID/result failure.'));emit_jit();return 2
  c.flush();t.flush()
 # Direct integer score validation is after every measured query has finished.
 post_errors=sum(true_errors(qi,docs,values) for qi,docs,values in retained)
 success=post_errors==0 and all(x['result_count']==wanted and x['position_count']==wanted and not any(x[k] for k in ['invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors','canonical_id_mismatches']) for x in checks)
 report=dict(success=success,samples=len(checks),true_score_errors=post_errors,canonical_mismatched_positions=sum(x['canonical_id_mismatches'] for x in checks),canonical_mismatched_groups=sum(bool(x['canonical_id_mismatches']) for x in checks),qualification='Primary uint64 scoring and canonical numeric-ID topk contract on the declared packed-selection input domain. All returned scores independently checked by integer CSR scoring after timing.',torch_allocator_peak_bytes=torch.cuda.max_memory_allocated(),torch_allocator_reserved_peak_bytes=torch.cuda.max_memory_reserved(),memory_qualification='Torch allocator peak since post-setup reset, including resident index and native query buffers; setup transients, driver/context and non-Torch allocations excluded. Not an Nsight application-peak measurement.')
 write_new(paths[2],report)
 emit_jit()
 print(json.dumps(report));return 0 if success else 2

if __name__=='__main__':
 a=arguments()
 if not os.environ.get('RESEARCH_RUNTIME_DIR'):raise RuntimeError('Use scripts/research.py run; unregistered retrieval is disabled')
 import torch
 with torch.inference_mode():raise SystemExit(main(a))
