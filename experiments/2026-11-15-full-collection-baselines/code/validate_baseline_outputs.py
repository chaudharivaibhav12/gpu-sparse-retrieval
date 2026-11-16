"""Validate every output/sample against a fixed independent reference."""
import csv,json,math
from pathlib import Path
import numpy as np
from prepare_reference import check_returned_scores

def reference(path,k):
    p=Path(path)
    if p.suffix=='.json':return {q:[tuple(map(int,pair)) for pair in values[:k]] for q,values in json.loads(p.read_text()).items()}
    result={}
    with p.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):result.setdefault(row['query_id'],[]).append((int(row['original_id']),int(row['score'])))
    return {q:values[:k] for q,values in result.items()}

def validate(label,csv_path,top_path,oracle_path,gsp_path,query_rows,cfg,allocation,run_id,oracle_sha,query_sha,gsp_sha,native_validation=None):
    k=cfg['k'] if 'k' in cfg else cfg['requested_k'];depth=min(k,cfg['docs'])
    qids=[q for q,t in query_rows];queries={q:dict(t) for q,t in query_rows}
    if len(set(qids))!=len(qids) or len(qids)!=cfg['query_count']:raise ValueError('query population')
    oracle=reference(oracle_path,depth)
    if set(qids)-set(oracle) or any(len(oracle[q])!=depth for q in qids):raise ValueError('oracle coverage/depth')
    wanted=[(phase,r,q) for phase,count in [('warmup',cfg['warmup_rounds']),('measured',cfg['repeats'])] for r in range(count) for q in qids]
    samples=[];latencies=[]
    with Path(csv_path).open(newline='') as f:
        for row in csv.DictReader(f):
            key=(row['phase'],int(row['round']),row['query_id']);samples.append(key)
            wall=float(row['query_wall_ms']);span=float(row['gpu_span_ms'])
            if not math.isfinite(wall) or wall<=0 or not math.isfinite(span) or span<0:raise ValueError('invalid timing')
            if int(row['docs'])!=cfg['docs'] or int(row['k'])!=k:raise ValueError('sample operating point')
            if label in ['range','csr'] and int(row['device_allocation_bytes'])!=allocation:raise ValueError('tracked allocation accounting')
            if label=='range':
                checks=dict(range_size=1024,ranges=cfg['ranges'],score_threads=256,merge_threads=256,index_posting_bytes=cfg['index_posting_bytes'],result_transfer_bytes=depth*24,query_transfer_bytes=1028)
                if any(int(row[key])!=value for key,value in checks.items()):raise ValueError('range sample/config accounting')
                expected='0' if cfg['verify_scores'] else 'not_checked'
                if row['ordered_id_mismatches']!=expected or row['result_score_mismatches']!=expected:raise ValueError('range internal verification state')
                if int(row['query_terms'])!=len(queries[row['query_id']]):raise ValueError('query term count')
            if label=='csr':
                checks=dict(pruned=0,order=0,bounds_bytes=0,seed_blocks=0,seed_docs=0,skipped_blocks=0,active_scored_blocks=(cfg['docs']+31)//32,result_transfer_bytes=depth*24,count_transfer_bytes=0)
                if any(int(row[key])!=value for key,value in checks.items()):raise ValueError('exhaustive schedule/accounting')
            if label=='mono':
                if row['variant']!='monolithic':raise ValueError('native variant label')
                if int(row['result_count'])!=depth or int(row['position_count'])!=depth:raise ValueError('native returned depth')
                for counter in ['invalid_positions','duplicate_results','noninteger_scores','score_sequence_errors','canonical_id_mismatches']:
                    if int(row[counter]):raise ValueError('native correctness counter: '+counter)
            if key[0]=='measured':latencies.append(dict(round=key[1],query_id=key[2],query_wall_ms=wall,gpu_span_ms=span))
    if samples!=wanted:raise ValueError('missing/duplicate/reordered samples')
    groups={};scores=0;group_order=[]
    id_map=np.memmap(gsp_path,dtype='<u8',mode='r',offset=24,shape=(cfg['docs'],)) if label=='mono' and cfg['docs'] else None
    with Path(top_path).open(newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            key=(row['phase'],int(row['round']),row['query_id'])
            if key not in groups:group_order.append(key)
            values=groups.setdefault(key,[])
            if int(row['rank'])!=len(values)+1:raise ValueError('rank sequence')
            if label=='csr' and int(row['pruned']):raise ValueError('unexpected pruned top result')
            if label=='mono':
                position=int(row['internal_doc_position'])
                if row['variant']!='monolithic' or not 0<=position<cfg['docs'] or int(id_map[position])!=int(row['original_id']):raise ValueError('native internal/original ID map')
            values.append((int(row['original_id']),int(row['score'])));scores+=1
    if set(groups)-set(wanted):raise ValueError('unexpected top groups')
    if group_order!=(wanted if depth else []):raise ValueError('top group order/coverage')
    for key in wanted:
        if groups.get(key,[])!=oracle[key[2]]:raise ValueError('canonical ID/score mismatch: '+str(key))
    # Every raw result was compared above; independently recompute unique
    # query/document scores once, avoiding duplicate work over repeats.
    checked=check_returned_scores(gsp_path,queries,{q:oracle[q] for q in qids}) if depth else 0
    if native_validation:
        native=json.loads(Path(native_validation).read_text())
        if not native['success'] or native['true_score_errors'] or native['canonical_mismatched_positions'] or native['samples']!=len(wanted):raise ValueError('native post-validation failed')
    report=dict(success=True,run_id=run_id,queries=len(qids),groups=len(wanted),scores_compared=scores,independent_scores_checked=checked,ordered_id_mismatches=0,score_mismatches=0,oracle_sha256=oracle_sha,query_sha256=query_sha,canonical_gsp_sha256=gsp_sha,internal_verify_scores=cfg['verify_scores'],qualification='Every declared raw sample/top-k matched the frozen reference; distinct query/document scores independently recomputed with Python integers. C++ internal verification state remains explicit; not_checked is not relabeled as an internal pass.')
    return report,latencies
