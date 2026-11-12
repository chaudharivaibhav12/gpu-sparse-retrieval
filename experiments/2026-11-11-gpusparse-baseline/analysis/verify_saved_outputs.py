"""Validate saved evidence offline. Requires NumPy/SciPy; never imports Torch."""
import csv,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from gsp_input import load_index,read_queries,result_score
RUN=ROOT/'results/2026-11-11-gpusparse-baseline/20261111-cuda5-native-01'

def verify():
    manifest=json.loads((RUN/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        p=ROOT/name
        assert p.resolve().is_relative_to(ROOT.resolve())
        assert p.stat().st_size==value['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256'],name
    attempts=json.loads((RUN/'attempts.json').read_text());groups=returned=0
    assert len(attempts)==12
    for attempt in attempts:
        tag=attempt['tag'];state=attempt['status']
        if state=='SPEC_VIOLATION':
            assert set(attempt['runtime_delta'])=={'config.block_pl'}
            assert not list((RUN/'raw').glob(tag+'*'))
            continue
        assert not attempt['runtime_delta']
        for item in attempt['inputs']:
            assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
        with (RUN/'raw'/(tag+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        with (RUN/'raw'/(tag+'.tsv')).open(newline='') as f:tops=list(csv.DictReader(f,delimiter='\t'))
        if state=='FAILED':
            assert not rows and not tops and attempt['exit_code']==1
            assert "'constexpr' object has no attribute 'to'" in (RUN/'logs'/(tag+'-stderr.txt')).read_text()
            continue
        assert state=='EXECUTED' and attempt['exit_code']==0
        runtime=json.loads((RUN/'logs'/(tag+'-runtime.json')).read_text())['config']
        paths={x['role']:ROOT/x['path'] for x in attempt['inputs']}
        index=load_index(paths['input-0']);queries=dict(read_queries(paths['input-1'],index['V'],runtime['query_count']))
        oracle=json.loads(paths['input-2'].read_text());k=min(runtime['requested_k'],index['N'])
        assert len(rows)==5 and [x['query_id'] for x in rows]==list(queries)
        assert all(x['phase']=='measured' and x['round']=='0' and x['variant']==runtime['variant'] for x in rows+tops)
        assert not {x['query_id'] for x in tops}-queries.keys()
        for row in rows:
            qid=row['query_id'];terms=queries[qid]
            expected=sorted([(int(index['ids'][d]),result_score(index,d,terms)) for d in range(index['N'])],key=lambda x:(-x[1],x[0]))[:k]
            assert expected==[tuple(x) for x in oracle[qid][:k]]
            actual=[x for x in tops if x['query_id']==qid]
            assert len(actual)==int(row['result_count'])==int(row['position_count'])==k
            assert [int(x['rank']) for x in actual]==list(range(1,k+1))
            assert [float(x['score']) for x in actual]==[x[1] for x in expected]
            assert len({x['internal_doc_position'] for x in actual})==k
            for x in actual:
                pos=int(x['internal_doc_position']);assert 0<=pos<index['N']
                assert int(x['original_id'])==int(index['ids'][pos])
                assert float(x['score'])==result_score(index,pos,terms);returned+=1
            for field in ['invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors','true_score_errors']:assert row[field]=='0'
            groups+=1
    assert groups==50 and returned==5
    print(json.dumps(dict(verified_groups=groups,returned_rows=returned,timing_processes=0)))

if __name__=='__main__':verify()
