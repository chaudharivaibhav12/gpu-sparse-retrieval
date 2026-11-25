"""GPUSparse's chunked batched scorer on the team's uint8 index, batch B, any k (pinned release 92bb87b; kernels unchanged).

canonical  the released scatter kernel and compaction kernel; only the selection is ours: packed int64 keys
           (score, then smaller original ID), so the answer is the team's exact top-k. Compaction keeps score >= tau, so
           ties with the k-th score are never dropped; a query whose candidates overflow the buffer is redone in full.
native     the released triton_chunked_score_fused, called as is. Ties are arbitrary: checked on scores, not on IDs.
Defaults of the release throughout: doc_chunk 131072, BLOCK_PL 128, 8192 candidates. FP32 sums are exact integers in the
guarded domain (guarded_domain.py). Timing of one batch: query padding and upload through host scores and original IDs.
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE), str(HERE / "gsk")]
REV = "92bb87b36f10e78ab39c81e9db2412dbb7a3b650"
SHA = {"triton_kernel_chunked.py": "444457e5ba207a0cf231693fdf6a3223c716fac1b4a19f03d59c5ef3adcdb761", "fused_topk.py": "e0ac84cc83a847267ce855aab7965326df8e68b6125d953e43a42b3eb7a24546"}
CHUNK, BLOCK_PL, CAND, BLOCK = 131072, 128, 8192, 1024
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_native(src):
    for n, h in SHA.items(): assert digest(Path(src) / n) == h, n
    sys.path.insert(0, str(src)); import triton_kernel_chunked as ck, fused_topk as ft
    return ck, ft


def canonical_fused(native, index, ti, tw, k, bounds, chunk=CHUNK, cand=CAND):
    import triton
    ck, ft = native; B, T = ti.shape; N = index.num_docs; dev = index.device; k = min(k, N)
    key = lambda s, r: (s.to(torch.int64) << 32) | (4294967295 - r.to(torch.int64))
    buf = torch.zeros(B, min(chunk, N), device=dev, dtype=torch.float32); run_k = run_p = None
    for c in range(bounds.shape[0] - 1):
        c0 = c * chunk; n = min(chunk, N - c0); view = buf[:, :n]; view.zero_(); kk = min(k, n)
        ck._chunked_scatter_add_kernel[(B, T)](index.doc_ids, index.scores, bounds[c], bounds[c + 1], ti, tw, view, c0, n, view.stride(0), T, index.vocab_size, BLOCK_PL=BLOCK_PL)
        ranks = index.id_ranks[c0:c0 + n]
        if run_k is None or run_k.shape[1] < k: s, p = torch.topk(key(view, ranks), kk, dim=1)
        else:
            m = max(cand, kk); tau = (run_k[:, -1] >> 32).to(torch.float32).contiguous()
            idx = torch.zeros((B, m), device=dev, dtype=torch.int32); val = torch.full((B, m), float("-inf"), device=dev); count = torch.zeros(B, device=dev, dtype=torch.int32)
            ft._compact_above_threshold_kernel[(B, triton.cdiv(n, BLOCK))](view, tau, idx, val, count, view.stride(0), n, m, BLOCK=BLOCK)
            used = torch.arange(m, device=dev)[None] < count[:, None]
            keys = torch.where(used, key(torch.where(used, val, 0.0), ranks[idx.to(torch.int64)]), -1)
            s, i = torch.topk(keys, kk, dim=1); p = torch.gather(idx.to(torch.int64), 1, i)
            over = count > m
            if bool(over.any()):
                rows = torch.nonzero(over, as_tuple=True)[0]; s[rows], p[rows] = torch.topk(key(view[rows], ranks), kk, dim=1)
        p = p + c0
        if run_k is not None: s, p = torch.cat([run_k, s], 1), torch.cat([run_p, p], 1)
        run_k, sel = torch.topk(s, min(k, s.shape[1]), dim=1); run_p = torch.gather(p, 1, sel)
    return run_k >> 32, run_p


def padded(rows, width):
    ids = np.full((len(rows), width), -1, np.int32); w = np.zeros((len(rows), width), np.float32)
    for i, (_, terms) in enumerate(rows):
        for j, (t, x) in enumerate(terms): ids[i, j] = t; w[i, j] = x
    return ids, w


def main():
    ap = argparse.ArgumentParser()
    for n in ("index", "queries", "reference", "native-repo", "csv", "top", "validation", "mode"): ap.add_argument("--" + n, required=True)
    for n in ("k", "batch", "repeats", "warmups"): ap.add_argument("--" + n, type=int, required=True)
    a = ap.parse_args(); assert a.mode in ("canonical", "native") and a.k >= 1 and a.batch >= 1
    import scipy, triton
    from gsp_input import load_index, read_queries, result_score
    from guarded_domain import query_bound
    from storage_domain import checked_u8
    from runtime_gate import runtime_report
    src = Path(a.native_repo) / "src"
    assert subprocess.check_output(["git", "-C", a.native_repo, "rev-parse", "HEAD"], text=True).strip() == REV
    native = load_native(src)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    ix = load_index(a.index); queries = read_queries(a.queries, ix["V"], 2 ** 63 - 1); query_bound(queries); ref = {}
    for r in csv.DictReader(open(a.reference, newline=""), delimiter="\t"): ref.setdefault(r["query_id"], []).append((int(r["original_id"]), int(r["score"])))
    ref = {q: v[:a.k] for q, v in ref.items()}; assert all(len(ref[q]) == a.k for q, _ in queries)
    csc = ix["csc"]; dev = torch.device("cuda", 0); N = ix["N"]; width = max(len(t) for _, t in queries)
    g = SimpleNamespace(device=dev, num_docs=N, vocab_size=ix["V"], doc_ids=torch.from_numpy(csc.indices.astype(np.int32)).to(dev), scores=torch.from_numpy(checked_u8(csc.data)).to(dev),
                        offsets=torch.from_numpy(csc.indptr[:-1].astype(np.int64)).to(dev), lengths=torch.from_numpy(np.diff(csc.indptr).astype(np.int32)).to(dev))
    order = np.argsort(ix["ids"], kind="stable"); ranks = np.empty(N, np.int32); ranks[order] = np.arange(N, dtype=np.int32); g.id_ranks = torch.from_numpy(ranks).to(dev)
    bounds = native[0].build_chunk_boundaries(g, CHUNK); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    cfg = dict(mode="gpusparse-chunked-" + a.mode, docs=N, nnz=ix["nnz"], query_count=len(queries), query_tensor_width=width, requested_k=a.k, batch_size=a.batch, doc_chunk=CHUNK, block_pl=BLOCK_PL, max_candidates=CAND,
               repeats=a.repeats, warmup_rounds=a.warmups, native_revision=REV, native_sha256=SHA, driver_sha256=digest(__file__), gpu_name=torch.cuda.get_device_name(0), torch_version=str(torch.__version__), triton_version=triton.__version__,
               numpy_version=np.__version__, scipy_version=scipy.__version__, torch_cuda_version=torch.version.cuda, document_gpu_dtype=str(g.scores.dtype), cpu_affinity=",".join(map(str, sorted(os.sched_getaffinity(0)))), torch_num_threads=torch.get_num_threads())
    runtime_report(cfg, dict(contract="GSP-CAP32-F32GUARDED-NUMERIC-TIES-v1" if a.mode == "canonical" else "score-exact-arbitrary-ties", timing="padding-and-upload-through-host-scores-and-original-ids"))
    kept, good, diff = [], True, 0
    with open(a.csv, "x", newline="") as c, open(a.top, "x", newline="") as t:
        w = csv.writer(c, lineterminator="\n"); w.writerow("phase round batch_index batch_size batch_wall_ms".split())
        t.write("phase\tround\tquery_id\trank\toriginal_id\tscore\n")
        for phase, n in (("warmup", a.warmups), ("measured", a.repeats)):
            for rnd in range(n):
                for b in range(0, len(queries), a.batch):
                    rows = queries[b:b + a.batch]; torch.cuda.synchronize(); t0 = time.perf_counter_ns()
                    hi, hw = padded(rows, width); ti, tw = torch.from_numpy(hi).to(dev), torch.from_numpy(hw).to(dev)
                    if a.mode == "canonical": s, p = canonical_fused(native, g, ti, tw, a.k, bounds)
                    else: s, p = native[1].triton_chunked_score_fused(g, ti, tw, top_k=a.k, doc_chunk=CHUNK, chunk_bounds=bounds)
                    s, p = s.cpu().numpy(), p.cpu().numpy(); ids = ix["ids"][p]
                    ms = (time.perf_counter_ns() - t0) / 1e6
                    w.writerow([phase, rnd, b // a.batch, len(rows), ms])
                    if a.mode == "canonical":
                        for i, (q, _) in enumerate(rows): t.write("".join("%s\t%d\t%s\t%d\t%d\t%d\n" % (phase, rnd, q, j + 1, d, x) for j, (d, x) in enumerate(zip(ids[i].tolist(), s[i].tolist()))))
                    else: kept.append((b, s, p, ids))
    if a.mode == "native":  # scores equal the oracle's; every ID above the k-th score is the oracle's; tied IDs are verified by rescoring
        cache = {}
        for b, s, p, ids in kept:
            for i, (q, terms) in enumerate(queries[b:b + a.batch]):
                want = ref[q]; got = s[i].tolist(); good &= got == [float(x) for _, x in want] and len(set(ids[i].tolist())) == a.k
                last = want[-1][1]; have = ids[i].tolist(); known = {d: x for d, x in want}
                good &= {d for d, x in want if x > last} == {d for d, x in zip(have, got) if x > last}
                diff += sum(d != e for d, (e, _) in zip(have, want))
                for d, x, pos in zip(have, got, p[i].tolist()):
                    if d not in known:
                        if (b, i, pos) not in cache: cache[b, i, pos] = result_score(ix, pos, terms)
                        good &= cache[b, i, pos] == x
                    else: good &= known[d] == x
    json.dump(dict(success=bool(good), mode=a.mode, id_positions_different_from_canonical=diff, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20), open(a.validation, "x"), indent=1)
    return 0 if good else 2


if __name__ == "__main__":
    if not os.environ.get("RESEARCH_RUNTIME_DIR"): raise SystemExit("run through code/run.py (runtime gate)")
    with torch.inference_mode(): raise SystemExit(main())
