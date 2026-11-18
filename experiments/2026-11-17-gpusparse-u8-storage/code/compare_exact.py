"""Claim-time system comparison with narrowly certified exact U32/U64 sums.

Keep the runner's original BLOCKED comparison, then append an explicit,
domain-qualified comparison. Native float/unspecified-tie contracts cannot use
this adapter. No global research-policy or runner relaxation is performed.
"""
import json

U64='GSP-CAP32-U64-NUMERIC-TIES-v1'
U32='GSP-CAP32-U32GUARDED-NUMERIC-TIES-v1'

def require(condition,message):
    if not condition:raise ValueError(message)

def roles(run):return {x['role']:x['sha256'] for x in run['inputs']}

def point(config):
    k=config['k'] if 'k' in config else config['requested_k']
    return tuple(config[x] for x in ['docs','query_count','batch_size','gpu_id','warmup_rounds','repeats','cpu_affinity'])+(k,)

def compare_certified(ledger,left_id,right_id,allowed,reason,certificate_id,target_uuid):
    from research import compare
    require(isinstance(allowed,dict) and all(allowed.values()) and bool(reason),'explicit comparison reasons required')
    ledger.verify();left=ledger.get(left_id)['payload'];right=ledger.get(right_id)['payload']
    require(left['status']==right['status']=='EXECUTED','incomplete comparison run')
    require(left['split']==right['split']=='calibration','only declared calibration comparison supported')
    lc=left['runtime']['config'];rc=right['runtime']['config'];lb=left['runtime']['behavior'];rb=right['runtime']['behavior']
    require(point(lc)==point(rc),'operating point/warmups/repeats/CPU affinity differ')
    require(left['timing_boundary']==right['timing_boundary'],'timing boundary differs')
    for role in ['canonical_gsp','queries','query_split']:
        require(role in roles(left) and roles(left)[role]==roles(right).get(role),'unmatched '+role)
    for run in [left,right]:require(target_uuid in run['environment']['gpu']['stdout'],'GPU UUID differs/missing')
    original=compare(ledger,left_id,right_id,allowed,reason)
    mismatch=set(original['payload']['contract_mismatches'])
    require(not original['payload']['unexplained'],'undeclared comparison deltas')
    require(mismatch<={'behavior.contract','behavior.arithmetic'},'unresolved semantic mismatch')
    from guarded_domain import CONTRACT as F32,ARITHMETIC as F32ARITHMETIC
    permitted={U32:'uint32-guarded-exact-widened-uint64',U64:'uint64',F32:F32ARITHMETIC}
    require(lb['contract'] in permitted and rb['contract'] in permitted,'unapproved arithmetic pair')
    for b in [lb,rb]:
        require(b['arithmetic']==permitted[b['contract']],'wrong physical arithmetic')
    for key in ['document_impacts','query_policy','tie_order','eligibility']:
        require(lb[key]==rb[key],'scoring/tie/eligibility differs')
    certificate=ledger.get(certificate_id);require(certificate['kind']=='observation','certificate must be observed')
    proof=certificate['payload'];require(proof['certificate_kind']=='guarded-integer-domain-u32-u64-fp32-v1','wrong domain certificate')
    require(proof['docs']==8841823 and proof['query_count']==500 and proof['k']==10 and proof['batch_size']==1,'wrong frozen scope')
    validation=ledger.get(proof['index_validation_ref'])['payload']['result']
    require(validation['success'] and validation['docs']==8841823 and validation['all_document_ids_replayed']==8841823,'full input validation missing')
    require(validation['maximum_term_document_multiplicity']==proof['maximum_term_document_multiplicity']==1,'unique term multiplicity required')
    bound=255*proof['max_query_weight_sum']
    require(0<=bound<2**24 and bound==proof['max_conservative_score_bound'],'unsafe FP32integer domain')
    require(proof['unique_cap32_query_terms_verified'] is True and 0<=proof['max_query_terms']<=128,'query audit missing')
    require(proof['canonical_gsp_sha256']==roles(left)['canonical_gsp'] and proof['queries_sha256']==roles(left)['queries'],'domain inputs differ')
    oracle=ledger.get(proof['oracle_run'])['payload']
    require(oracle['status']=='EXECUTED' and oracle['runtime']['config']['mode']=='cpu-sequential' and oracle['runtime']['config']['docs']==8841823 and oracle['runtime']['config']['query_count']==500 and oracle['runtime']['config']['k']==10 and oracle['runtime']['behavior']['contract']==U64,'oracle scope/contract differs')
    require(roles(oracle)['index']==proof['canonical_gsp_sha256'] and roles(oracle)['queries']==proof['queries_sha256'],'oracle inputs differ')
    require(any(x['sha256']==proof['oracle_sha256'] for x in oracle['artifacts']),'oracle bytes unpreserved')
    for run in [left,right]:
        c=run['runtime']['config'];b=run['runtime']['behavior']
        require(c['docs']==8841823 and c['query_count']==500 and c['batch_size']==1,'runtime scope differs')
        require(roles(run)['reference']==proof['oracle_sha256'] and roles(run)['query_split']==roles(oracle)['query_split'] and oracle['split']=='calibration','reference/split differs')
        require(c['max_conservative_score_bound']==bound and c['max_query_terms']==proof['max_query_terms'],'runtime domain differs')
        if b['contract']==U32:
            require(c['score_accumulator_bits']==32 and c['returned_score_bits']==64 and c['score_limit_exclusive']==2**32-1 and c['max_term_document_multiplicity']==1 and c['source_gsp_sha256']==proof['canonical_gsp_sha256'],'range width/source differs')
        else:
            require(c['selection_score_limit']==2**31-1 and bound<c['selection_score_limit'],'packed selector domain differs')
            if b['contract']==F32:
                require(c['mode'] in ['gpusparse-fp32guarded-canonical','gpusparse-fp32guarded-u8-canonical'] and c['accumulator_dtype']=='torch.float32' and c['query_gpu_dtype']=='torch.float32' and c['score_output_dtype']=='torch.int64','FP32 physical dtype differs')
                require(c['document_gpu_dtype']==('torch.uint8' if c['mode']=='gpusparse-fp32guarded-u8-canonical' else 'torch.float32') and c['selection_key_dtype']=='torch.int64' and c['selection_score_bits']==31 and c['selection_id_bits']==32 and c['max_supported_terms']==128,'FP32 input/selector domain differs')
                require(c['floating_score_limit_exclusive']==2**24 and c['native_scoring_reused'] is True,'FP32guard/native reuse missing')
                require(c['native_monolithic_sha256']==proof['native_monolithic_sha256']==roles(run)['native_scoring_source'],'native source not bound')
                require(c['selector_module_sha256']==proof['selector_module_sha256'],'canonical selector source differs')
                if c['mode']=='gpusparse-fp32guarded-u8-canonical':
                    require(c['document_storage_bytes']==1 and c['document_load_conversion']=='uint8-to-fp32-before-product','compact storage metadata differs')
                    require(b.get('document_storage')=='uint8' and b.get('document_load_conversion')=='uint8-to-fp32-before-product','compact storage behavior differs')
                    require(c['index_gpu_bytes']==c['nnz']*5+c['vocab']*12+c['docs']*4,'compact allocation model differs')

            else:require(c['mode']=='gpusparse-u64-canonical' and c['accumulator_dtype']=='torch.uint64','U64dtype differs')
    # Each measured binary/adapter must have passed full-corpus correctness with
    # this same input; a certificate for another implementation is insufficient.
    for measured in [left,right]:
        matches=[]
        for entry in proof['correctness']:
            run=ledger.get(entry['run'])['payload']
            if run['build']!=measured['build'] or run['runtime']['config']['mode']!=measured['runtime']['config']['mode']:continue
            require(run['status']=='EXECUTED' and run['split']=='calibration','invalid correctness run')
            require(target_uuid in run['environment']['gpu']['stdout'],'correctness GPU differs')
            cconfig=run['runtime']['config'];mconfig=measured['runtime']['config']
            exclude={'verify_scores','warmup_rounds','repeats'}
            require({k:v for k,v in cconfig.items() if k not in exclude}=={k:v for k,v in mconfig.items() if k not in exclude},'correctness algorithm/settings differ')
            cb=run['runtime']['behavior'];mb=measured['runtime']['behavior']
            require({k:v for k,v in cb.items() if k!='cpu_validation'}=={k:v for k,v in mb.items() if k!='cpu_validation'},'correctness behavior differs')
            require(roles(run)['canonical_gsp']==proof['canonical_gsp_sha256'] and roles(run)['queries']==proof['queries_sha256'],'correctness input differs')
            require(run['runtime']['config']['docs']==proof['docs'] and run['runtime']['config']['query_count']==proof['query_count'],'correctness corpus/query scope')
            artifact=entry['validation'];path=ledger.root/artifact['blob']
            from research import digest
            require(artifact['blob']=='research/evidence/'+artifact['sha256'] and artifact in proof['artifacts'],'validation blob not canonically preserved')
            require(digest(path)==artifact['sha256'],'changed validation artifact')
            report=json.loads(path.read_text())
            require(report['run_id']==entry['run'],'validation belongs to another run')
            require(report['success'] and report['queries']==proof['query_count'] and report['groups']==proof['query_count'],'missing complete correctness groups')
            require(report['ordered_id_mismatches']==report['score_mismatches']==0 and report['independent_scores_checked']==proof['query_count']*10,'canonical correctness not certified')
            require(report['oracle_sha256']==proof['oracle_sha256'],'oracle identity differs')
            matches.append(entry['run'])
        require(matches,'no same-build full-corpus correctness evidence')
    payload=dict(original['payload']);payload.update(status='REVIEWABLE',contract_mismatches=[],observed_contract_differences=sorted(mismatch),domain_equivalence_certificate=certificate_id,original_blocked_comparison=original['id'],qualification='Integer sums and canonical outputs coincide only on the verified bounded nonnegative integer domain, including guarded FP32 exact-integer arithmetic. Physical accumulator widths remain distinct. All implementation/timing/resource differences remain disclosed; no unspecified-tie or arbitrary-float/general-input guarantee.')
    return ledger.event('comparison',payload,[left_id,right_id,original['id'],certificate_id])
