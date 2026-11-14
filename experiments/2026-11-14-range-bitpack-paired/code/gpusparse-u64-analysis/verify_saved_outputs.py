"""Verify the public scientific subset without the private recording infrastructure."""
import argparse,csv,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'code'))
from gsp_input import load_index,read_queries,result_score
from canonical_baseline import references,CONTRACT

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def validate_group(rows,reference):
    expected=[(rank,int(identity),int(score)) for rank,(identity,score) in enumerate(reference,1)]
    assert rows==expected,'Canonical ordered ID/score mismatch'

def verify(root,run):
    root=Path(root).resolve();run=Path(run).resolve();assert run.is_relative_to(root)
    manifest=read(run/'manifest.json');assert manifest['contract']==CONTRACT['contract']
    for item in manifest['artifacts']:
        path=root/item['path'];assert path.resolve().is_relative_to(root)
        assert path.is_file() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],item['path']
    source=manifest['source']
    for name,value in source['code_hashes'].items():assert digest(E/'code'/name)==value
    for name,value in source['source_hashes'].items():assert digest(root/name)==value
    attempts=read(run/'attempts.json');assert len(attempts)==86 and len({a['tag'] for a in attempts})==86
    statuses=Counter(a['status'] for a in attempts);assert statuses=={'EXECUTED':85,'SPEC_VIOLATION':1}
    candidates={};by_id={}
    for line in (run/'analysis/candidate-rows.jsonl').read_text().splitlines():
        row=json.loads(line);assert row['position'] not in candidates and row['original_id'] not in by_id
        assert len({t for t,v in row['terms']})==len(row['terms'])
        assert all(t>=0 and 0<=v<=255 for t,v in row['terms'])
        candidates[row['position']]=row;by_id[row['original_id']]=row
    provenance=read(run/'analysis/candidate-rows-manifest.json')
    assert provenance['rows']==len(candidates) and provenance['output_sha256']==digest(run/'analysis/candidate-rows.jsonl')
    fixtures={};score_cache={};group_count=score_count=0
    for attempt in attempts:
        tag=attempt['tag'];cfg=attempt['config'];behavior=attempt['behavior']
        assert all(behavior[k]==v for k,v in CONTRACT.items())
        paths={}
        for item in attempt['inputs']:
            path=root/item['public_path'] if item['public_path'] else None
            if path is not None:assert path.resolve().is_relative_to(root) and digest(path)==item['sha256']
            else:assert item['sha256']==provenance['index_sha256']
            paths[item['role']]=path
        if tag=='wrong-pack-threads':
            assert attempt['status']=='SPEC_VIOLATION' and set(attempt['runtime_delta'])=={'config.pack_threads'}
            assert not list((run/'raw').glob(tag+'*'));continue
        assert attempt['status']=='EXECUTED' and attempt['exit_code']==0 and not attempt['runtime_delta']
        gsp=cfg['mode']=='gpusparse-u64-canonical';depth=cfg['returned_depth'] if gsp else min(cfg['k'],cfg['docs'])
        index=None
        if paths['input-0'] is not None:
            path=paths['input-0']
            if str(path) not in fixtures:fixtures[str(path)]=load_index(path)
            index=fixtures[str(path)];vocab=index['V']
        else:vocab=cfg.get('vocab',provenance['vocabulary'])
        queries=read_queries(paths['input-1'],vocab,cfg['query_count']);weights={q:dict(t) for q,t in queries}
        reference=references(paths['input-2'],depth)
        if index is not None:
            for q,terms in queries:
                key=('fixture-oracle',str(paths['input-0']),q)
                if key not in score_cache:
                    truth=sorted([(int(identity),result_score(index,p,terms)) for p,identity in enumerate(index['ids'])],key=lambda x:(-x[1],x[0]))
                    score_cache[key]=truth
                assert reference[q]==score_cache[key][:depth]
        with (run/'raw'/(tag+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        expected=[(phase,str(r),q) for phase,n in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for r in range(n) for q,_ in queries]
        assert [(r['phase'],r['round'],r['query_id']) for r in rows]==expected
        for row in rows:
            value=float(row['query_wall_ms']);assert math.isfinite(value) and value>0
            if gsp:
                assert int(row['result_count'])==int(row['position_count'])==depth
                assert all(row[k]=='0' for k in ['invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors','canonical_id_mismatches'])
                assert row['true_score_errors']==('0' if cfg['verify_scores'] else 'post_validation')
            else:
                assert row['pruned']=='1' and row['precision']=='0'
                assert all(row[k]==('0' if cfg['verify_scores'] else 'not_checked') for k in ['evaluated_score_mismatches','ordered_id_mismatches','result_score_mismatches','unsafe_pruned_documents','invalid_upper_bounds','invalid_schedule','threshold_mismatches'])
                blocks=(cfg['docs']+cfg['block_size']-1)//cfg['block_size']
                assert sum(int(row[k]) for k in ['seed_blocks','active_scored_blocks','skipped_blocks'])==blocks
        actual={}
        with (run/'raw'/(tag+'.tsv')).open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                q=row['query_id'];identity=int(row['original_id']);score=int(row['score']);key=(row['phase'],row['round'],q)
                actual.setdefault(key,[]).append((int(row['rank']),identity,score));score_count+=1
                if index is not None:
                    position=int(row['internal_doc_position']);assert 0<=position<index['N'] and int(index['ids'][position])==identity
                else:
                    data=by_id[identity]
                    if gsp:assert data['position']==int(row['internal_doc_position'])
                memo=('returned-score',str(paths['input-0']),q,identity)
                if memo not in score_cache:
                    if index is not None:
                        truth=result_score(index,position,list(weights[q].items()))
                    else:
                        truth=sum(value*weights[q].get(term,0) for term,value in data['terms'])
                    score_cache[memo]=truth
                assert score==score_cache[memo]
        for key in expected:validate_group(actual.get(key,[]),reference[key[2]])
        assert not set(actual)-set(expected);group_count+=len(rows)
        if gsp:
            report=read(run/'raw'/(tag+'-validation.json'))
            assert report['success'] and report['samples']==len(rows)
            assert report['true_score_errors']==report['canonical_mismatched_groups']==report['canonical_mismatched_positions']==0
    assert group_count==105860
    comparisons=read(run/'analysis/comparisons.json');assert len(comparisons)==6
    assert all(c['status']=='REVIEWABLE' and not c['contract_mismatches'] and not c['unexplained'] for c in comparisons)
    return dict(success=True,validated_groups=group_count,returned_scores=score_count,attempt_statuses=dict(statuses),
        qualification='Public artifact integrity, canonical output and direct integer score replay. Generated binaries/JIT caches and original instrumentation records remain privately preserved; bulk-index membership cannot be authenticated from partial candidate rows alone.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=E.parents[1]);p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.repo,a.run)))
