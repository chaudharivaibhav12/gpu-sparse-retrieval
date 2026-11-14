"""Independent CPU resolution of the complete expected runtime state; no CUDA imports."""
import hashlib, struct
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expected(index, queries, reference, native_repo, adapter, environment, cpu,
             limit, k, block_pl, repeats, warmups, verify, native_hashes, native_revision):
    with Path(index).open('rb') as f:
        assert f.read(8) == b'GSPIDX01'
        n, vocab, nnz = struct.unpack('<IIQ', f.read(16))
    rows = [line.split() for line in Path(queries).read_text(encoding='utf-8').splitlines()[:limit]]
    lengths = [len(row)-1 for row in rows]
    bound = max((255*sum(int(token.split(':')[1]) for token in row[1:]) for row in rows), default=0)
    code = Path(adapter).parent
    return dict(mode='gpusparse-u64-canonical', variant='monolithic', docs=n, vocab=vocab, nnz=nnz,
        query_count=len(rows), max_query_terms=max(lengths, default=0), query_tensor_width=max(1,max(lengths,default=0)),
        requested_k=k, returned_depth=min(k,n), batch_size=1, max_supported_k=128,
        gpu_id=0, gpu_name=environment['gpu_name'], compute_capability=environment['compute_capability'],
        block_pl=block_pl, index_gpu_bytes=nnz*8+vocab*12+n*4, canonical_rank_bytes=n*4,
        dense_score_bytes=n*8, packed_key_bytes=n*8, index_path=str(index), queries_path=str(queries),
        reference_path=str(reference), native_repo=str(Path(native_repo).resolve()), repeats=repeats,
        warmup_rounds=warmups, verify_scores=verify, max_conservative_score_bound=bound,
        selection_score_limit=2**31-1, selection_score_bits=31, selection_id_bits=32, pack_threads=256,
        native_revision=native_revision, native_monolithic_sha256=native_hashes['triton_kernel.py'],
        adapted_kernel_sha256=digest(code/'u64_monolithic.py'), adapter_sha256=digest(adapter),
        input_adapter_sha256=digest(code/'gsp_input.py'), runtime_gate_sha256=digest(code/'runtime_gate.py'),
        torch_version=environment['torch_version'], triton_version=environment['triton_version'],
        numpy_version=environment['numpy_version'], scipy_version=environment['scipy_version'],
        torch_cuda_version=environment['torch_cuda_version'], document_gpu_dtype='torch.float32',
        posting_gpu_dtype='torch.int32', offset_gpu_dtype='torch.int64', length_gpu_dtype='torch.int32',
        rank_gpu_dtype='torch.int32', accumulator_dtype='torch.uint64', selection_key_dtype='torch.int64',
        score_output_dtype='torch.int64', cpu_affinity=str(cpu), torch_num_threads=1,
        torch_interop_threads=1, grad_enabled=False)

def marks(verify):
    return dict(contract='GSP-CAP32-U64-NUMERIC-TIES-v1', arithmetic='uint64',
        document_impacts='uint8-exact', query_policy='BMP-f32-ceil-cap32',
        tie_order='score_desc_original_numeric_id_asc', eligibility='all_documents',
        scoring='adapted-gpusparse-monolithic-u64', selection='unique-packed-int64-key-torch-topk',
        canonical_adaptation='u64-products-and-sums;original-uint64-IDrank;31scorebits+32tie_bits',
        query_terms='all-retained-global-padding-minus1', posting_order='ascending-internal-document-position',
        integer_domain_guard='uint8-impacts;unique-cap32-terms;255*sumweights<=2^31-1',
        query_validation='all-file-rows-before-limit', cpu_validation='perquery-plus-post' if verify else 'post-timing-all-outputs',
        timing='query-padding-and-upload-through-adapted-call-pack-select-decode-resultcopy-and-originalIDmap',
        index_setup='excluded;CSR-toCSC+originalIDrank+GPU-upload',
        native_source='derived-from-pinned-monolithic;u64-products-accumulator+canonical-packed-selector',
        requested_k_domain='positive-request-through128;empty-index-returns-zero-depth',
        key_order='score-desc-original-numeric-ID-asc;unique-positive-int64;no-key-overflow')
