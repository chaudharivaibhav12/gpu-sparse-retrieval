"""Fixtures: check_gs.py NATIVE_SRC. The two batched modes and the k<=1000 monolithic copy against brute force (dense ties, small
chunks, overflow of the candidate buffer, zero scores, unsorted original IDs, short last chunk); and that gsk/ differs from the
pinned adapter only in its k limit."""
import difflib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, scipy.sparse as sp, torch
H = Path(__file__).resolve().parent; sys.path[:0] = [str(H), str(H / "gsk")]
import gs_batch as G, fp32_monolithic as M
A = H.parents[1] / "2026-11-17-gpusparse-u8-storage/code"
assert (A / "guarded_domain.py").read_bytes() == (H / "guarded_domain.py").read_bytes(); n = 0
for f in sorted((A / "fp32").glob("*.py")):
    a, b = f.read_text(), (H / "gsk" / f.name).read_text()
    for s in ("not 1<=a.k<=%d", "max_supported_k=%d", "positive-request-through%d", "assert 1<=top_k<=%d"): n += a.count(s % 128); a = a.replace(s % 128, s % 1000) if f.name in ("canonical_baseline.py", "expected_config.py", "fp32_monolithic.py") else a
    assert a == b, f.name
print("adapter copy differs only in the k limit")
src = Path(sys.argv[1]); native = G.load_native(src); M.load_native(src / "triton_kernel.py"); rng = np.random.default_rng(7); runs = 0; over = 0
orig = native[1]._compact_above_threshold_kernel
with torch.inference_mode():
    for N, V, dens, hi, chunk, cand in ((700, 40, .3, 3, 64, 16), (2500, 60, .2, 2, 512, 64), (3000, 50, .05, 255, 256, 8192), (1500, 30, .02, 4, 1500, 32), (4097, 80, .4, 2, 1024, 1100), (900, 20, .01, 1, 128, 8)):
        for _ in range(3):
            d = (rng.random((N, V)) < dens) * rng.integers(0, hi + 1, (N, V)); csc = sp.csc_matrix(d.astype(np.uint8)); csc.sort_indices()
            ids = rng.permutation(N * 3)[:N].astype(np.uint64); rank = np.empty(N, np.int32); rank[np.argsort(ids)] = np.arange(N, dtype=np.int32); cu = lambda x: torch.from_numpy(x).cuda()
            g = SimpleNamespace(device=torch.device("cuda", 0), num_docs=N, vocab_size=V, doc_ids=cu(csc.indices.astype(np.int32)), scores=cu(csc.data.astype(np.uint8)), offsets=cu(csc.indptr[:-1].astype(np.int64)), lengths=cu(np.diff(csc.indptr).astype(np.int32)), id_ranks=cu(rank))
            rows = [("q%d" % i, [(int(t), int(rng.integers(1, 33))) for t in rng.choice(V, rng.integers(1, 12), replace=False)]) for i in range(9)] + [("z", [(0, 0)])]
            hi_, hw = G.padded(rows, 14); ti, tw = cu(hi_), cu(hw); bounds = native[0].build_chunk_boundaries(g, chunk)
            for k in (1, 10, 100, 600):
                exp = []
                for _, t in rows:
                    s = sum(d[:, a].astype(np.int64) * w for a, w in t); o = np.lexsort((ids, -s))[:k]; exp.append((s[o].tolist(), o.tolist()))
                s, p = G.canonical_fused(native, g, ti, tw, k, bounds, chunk, cand); s, p = s.tolist(), p.tolist()
                assert [(x, y) for x, y in zip(s, p)] == exp, ("canonical", N, chunk, cand, k)
                s, p = native[1].triton_chunked_score_fused(g, ti, tw, top_k=k, doc_chunk=chunk, chunk_bounds=bounds, max_cand=cand)
                assert k > chunk or s.tolist() == [list(map(float, x)) for x, _ in exp], ("native", N, chunk, cand, k)
                for i in range(len(rows)):
                    s, p = M.canonical_monolithic(g, ti[i:i + 1], tw[i:i + 1], top_k=k, block_pl=512)
                    assert (s.tolist()[0], p.tolist()[0]) == exp[i], ("mono", N, k)
                runs += 1
print("fixtures passed:", runs)
