"""Independent CPU expected-state resolution for the registered native adapter."""
import hashlib,struct
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expected(index,queries,reference,native_repo,adapter,environment,cpu,variant,limit,k,block_pl,doc_chunk,max_cand,repeats,warmups,verify,native_hashes,native_revision):
 with Path(index).open('rb') as f:
  assert f.read(8)==b'GSPIDX01';N,V,nnz=struct.unpack('<IIQ',f.read(16))
 lines=Path(queries).read_text(encoding='utf-8').splitlines()[:limit]
 lengths=[len(s.split())-1 for s in lines];weight_sums=[sum(int(t.split(':')[1]) for t in s.split()[1:]) for s in lines]
 chunked=variant!='monolithic';chunks=(N+doc_chunk-1)//doc_chunk if chunked else 0;wanted=min(k,N)
 config=dict(mode='gpusparse-native',variant=variant,docs=N,vocab=V,nnz=nnz,query_count=len(lines),max_query_terms=max(lengths,default=0),query_tensor_width=max(1,max(lengths,default=0)),requested_k=k,returned_depth=wanted,batch_size=1,gpu_id=0,gpu_name=environment['gpu_name'],compute_capability=environment['compute_capability'],block_pl=block_pl,doc_chunk=doc_chunk if chunked else 0,max_cand=max_cand if variant=='compacted' else 0,chunk_count=chunks,chunk_boundary_bytes=(chunks+1)*V*8 if chunked else 0,index_gpu_bytes=nnz*8+V*12,index_path=str(index),queries_path=str(queries),reference_path=str(reference),native_repo=str(Path(native_repo).resolve()),repeats=repeats,warmup_rounds=warmups,verify_scores=verify,max_conservative_score_bound=255*max(weight_sums,default=0),native_revision=native_revision,native_monolithic_sha256=native_hashes['triton_kernel.py'],native_chunked_sha256=native_hashes['triton_kernel_chunked.py'],native_compacted_sha256=native_hashes['fused_topk.py'],adapter_sha256=digest(adapter),input_adapter_sha256=digest(Path(adapter).with_name('gsp_input.py')),torch_version=environment['torch_version'],triton_version=environment['triton_version'],numpy_version=environment['numpy_version'],scipy_version=environment['scipy_version'],torch_cuda_version=environment['torch_cuda_version'],document_gpu_dtype='torch.float32',posting_gpu_dtype='torch.int32',offset_gpu_dtype='torch.int64',length_gpu_dtype='torch.int32',cpu_affinity=str(cpu),torch_num_threads=1,torch_interop_threads=1,grad_enabled=False,max_effective_cand=max(max_cand,min(wanted,doc_chunk,N-doc_chunk)) if variant=='compacted' and N>doc_chunk else 0,compaction_block=1024 if variant=='compacted' and N>doc_chunk else 0)
 config['runtime_gate_sha256']=digest(Path(adapter).with_name('runtime_gate.py'))
 config['compatibility_patch_id']='stride-int64-promotion-r1'
 config['compatibility_patch_sha256']=digest(Path(adapter).with_name('compat-r1.patch'))
 config['original_native_chunked_sha256']='444457e5ba207a0cf231693fdf6a3223c716fac1b4a19f03d59c5ef3adcdb761'
 return config

def marks(variant,verify):
 return dict(contract='GSP-CAP32-F32-NATIVE-TIES-v1',arithmetic='float32',document_impacts='uint8-exact',query_policy='BMP-f32-ceil-cap32',tie_order='score_desc_backend_unspecified_equal_scores',eligibility='all_documents',scoring='pinned-compat-r1-'+variant,selection='torch.topk-native-ties' if variant!='compacted' else 'threshold-compaction-plus-torch.topk-native-ties',canonical_adaptation='disabled',query_terms='all-retained-global-padding-minus1',posting_order='ascending-internal-document-position',integer_domain_guard='uint8-impacts;unique-cap32-terms;255*sumweights<=2^24',query_validation='all-file-rows-before-limit',cpu_validation='perquery-plus-post' if verify else 'post-timing-all-outputs',timing='query-padding-and-upload-through-native-call-resultcopy-and-originalIDmap',index_setup='excluded;CSR-toCSC+GPU-upload+native-chunk-boundaries',native_source='pinned-plus-disclosed-stride-int64-promotion-r1',candidate_overflow='full-chunk-topk-fallback' if variant=='compacted' else 'not-applicable',requested_k_domain='positive-request;empty-index-returns-zero-depth')
