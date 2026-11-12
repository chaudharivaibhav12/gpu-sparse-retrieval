"""Offline hash, top-k, native-tie and direct-integer-score replay; no Torch."""
import csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from gsp_input import load_index,read_queries,result_score
RUN=ROOT/'results/2026-11-11-gpusparse-baseline/20261111-cuda5-compat-r1-01'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def references(path,k):
    if path.suffix=='.json':return {q:[tuple(x) for x in values[:k]] for q,values in read(path).items()}
    groups={}
    with path.open(newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):groups.setdefault(r['query_id'],[]).append((int(r['original_id']),int(r['score'])))
    return {q:values[:k] for q,values in groups.items()}

def verify():
    manifest=read(RUN/'manifest.json')
    for name,value in manifest['files'].items():
        p=ROOT/name;assert p.resolve().is_relative_to(ROOT.resolve())
        assert p.stat().st_size==value['bytes'] and sha(p)==value['sha256'],name
    for name,value in manifest['source']['code_hashes'].items():assert sha(Path(__file__).resolve().parents[1]/'code'/name)==value
    attempts=read(RUN/'attempts.json');assert len(attempts)==133
    candidate_rows={}
    with (RUN/'analysis/candidate-rows.jsonl').open() as f:
        for line in f:
            row=json.loads(line);assert row['position'] not in candidate_rows;candidate_rows[row['position']]=row
    assert read(RUN/'analysis/candidate-rows-manifest.json')['output_sha256']==sha(RUN/'analysis/candidate-rows.jsonl')
    indexes={};score_cache={};groups=returned=canonical_positions=canonical_groups=positive=negative=0
    for attempt in attempts:
        tag=attempt['tag'];cfg=attempt['config']
        if attempt['status']=='SPEC_VIOLATION':
            assert tag=='wrong-block-pl' and set(attempt['runtime_delta'])=={'config.block_pl'}
            assert not list((RUN/'raw').glob(tag+'*'));negative+=1;continue
        assert attempt['status']=='EXECUTED' and attempt['exit_code']==0 and not attempt['runtime_delta'];positive+=1
        runtime=read(RUN/'logs'/(tag+'-runtime.json'))
        assert runtime['config']==cfg and runtime['behavior']==attempt['behavior']
        assert cfg['compatibility_patch_id']=='stride-int64-promotion-r1'
        inputs={}
        for item in attempt['inputs']:
            p=ROOT/item['public_path'] if item['public_path'] else None
            if p is None:assert item['sha256']=='a7249ab0fafcfb1bff754e960063c5982f92e509f40d5ebbfda6a06cbb3e3c37'
            else:assert sha(p)==item['sha256']
            inputs[item['role']]=p
        index_path=inputs['input-0']
        if index_path is not None:
            if index_path not in indexes:indexes[index_path]=load_index(index_path)
            index=indexes[index_path];vocab=index['V']
        else:index=None;vocab=28679
        queries=dict(read_queries(inputs['input-1'],vocab,cfg['query_count']));reference=references(inputs['input-2'],cfg['returned_depth'])
        wanted=[(phase,str(i),q) for phase,n in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for i in range(n) for q in queries]
        with (RUN/'raw'/(tag+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        assert [(r['phase'],r['round'],r['query_id']) for r in rows]==wanted
        for r in rows:
            assert r['variant']==cfg['variant'] and int(r['result_count'])==int(r['position_count'])==cfg['returned_depth']
            assert math.isfinite(float(r['query_wall_ms'])) and float(r['query_wall_ms'])>0
            for key in ['invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors']:assert r[key]=='0'
            assert r['true_score_errors']==('0' if cfg['verify_scores'] else 'post_validation')
        rows_by_key=dict(zip(wanted,rows))
        tops={}
        with (RUN/'raw'/(tag+'.tsv')).open(newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):tops.setdefault((r['phase'],r['round'],r['query_id']),[]).append(r)
        assert not set(tops)-set(wanted)
        changed_positions=0;changed_groups=set()
        for key in wanted:
            values=tops.get(key,[]);qid=key[2];assert len(values)==cfg['returned_depth']
            assert [int(r['rank']) for r in values]==list(range(1,len(values)+1))
            assert len({r['internal_doc_position'] for r in values})==len(values)
            assert [float(r['score']) for r in values]==[s for i,s in reference[qid]]
            for r in values:
                assert r['variant']==cfg['variant']
                pos=int(r['internal_doc_position']);score=float(r['score']);assert 0<=pos<cfg['docs'] and math.isfinite(score) and score.is_integer()
                if index is not None:
                    assert int(r['original_id'])==int(index['ids'][pos]);truth=result_score(index,pos,queries[qid])
                else:
                    candidate=candidate_rows[pos];assert int(r['original_id'])==candidate['original_id'];cache_key=(qid,pos)
                    if cache_key not in score_cache:
                        weights=dict(queries[qid]);score_cache[cache_key]=sum(int(w)*weights.get(int(t),0) for t,w in candidate['terms'])
                    truth=score_cache[cache_key]
                assert score==truth
                if int(r['original_id'])!=reference[qid][int(r['rank'])-1][0]:changed_positions+=1;changed_groups.add(key)
                returned+=1
            assert int(rows_by_key[key]['canonical_id_mismatches'])==sum(int(r['original_id'])!=reference[qid][int(r['rank'])-1][0] for r in values)
        validation=read(RUN/'raw'/(tag+'-validation.json'))
        assert validation['success'] and validation['true_score_errors']==0 and validation['samples']==len(wanted)
        assert changed_positions==validation['canonical_mismatched_positions'] and len(changed_groups)==validation['canonical_mismatched_groups']
        groups+=len(wanted);canonical_positions+=changed_positions;canonical_groups+=len(changed_groups)
    assert (positive,negative,groups,returned,canonical_positions,canonical_groups)==(132,1,210540,2132175,32258,2237)
    print(json.dumps(dict(success=True,executed=positive,intentional_rejection=negative,groups=groups,returned_scores=returned,native_tie_groups=canonical_groups)))

if __name__=='__main__':verify()
