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
    if original['payload']['status']=='REVIEWABLE':return original
    mismatch=set(original['payload']['contract_mismatches'])
    require(not original['payload']['unexplained'],'undeclared comparison deltas')
    require(mismatch and mismatch<={'behavior.contract','behavior.arithmetic'},'unresolved semantic mismatch')
    require({lb['contract'],rb['contract']}=={U32,U64},'no approved guarded-U32/U64 pair')
    for key in ['document_impacts','query_policy','tie_order','eligibility']:
        require(lb[key]==rb[key],'scoring/tie/eligibility differs')
    guarded=left if lb['contract']==U32 else right;wide=right if guarded is left else left
    require(guarded['runtime']['behavior']['arithmetic']=='uint32-guarded-exact-widened-uint64','wrong guarded arithmetic')
    require(wide['runtime']['behavior']['arithmetic']=='uint64','wrong wide arithmetic')
    gc=guarded['runtime']['config'];certificate=ledger.get(certificate_id)
    require(certificate['kind']=='observation','certificate must preserve observed evidence')
    proof=certificate['payload'];require(proof['certificate_kind']=='guarded-u32-u64-exact-domain-v1','unsupported certificate')
    require(proof['docs']==8841823 and proof['query_count']==500 and proof['k']==10 and proof['batch_size']==1,'certificate is not frozen full-collection point')
    require(point(gc)[-1]==proof['k'] and gc['batch_size']==proof['batch_size'],'certificate k/batch differs')
    require(proof['canonical_gsp_sha256']==roles(guarded)['canonical_gsp'] and proof['queries_sha256']==roles(guarded)['queries'],'certificate input identity')
    require(proof['docs']==gc['docs'] and proof['query_count']==gc['query_count'],'certificate scope')
    validation=ledger.get(proof['index_validation_ref'])['payload']['result']
    require(validation['success'] and validation['docs']==gc['docs'] and validation['all_document_ids_replayed']==gc['docs'],'missing full corpus validation')
    multiplicity=validation['maximum_term_document_multiplicity']
    require(multiplicity==proof['maximum_term_document_multiplicity']==gc['max_term_document_multiplicity'],'multiplicity proof')
    bound=255*proof['max_query_weight_sum']*multiplicity
    require(0<=bound<2**32-1 and bound==proof['max_conservative_score_bound']==gc['max_conservative_score_bound'],'unsafe/unverified score bound')
    require(0<=proof['max_query_terms']<=128 and proof['max_query_terms']==gc['max_query_terms'],'query term domain')
    require(gc['score_accumulator_bits']==32 and gc['returned_score_bits']==64 and gc['score_limit_exclusive']==2**32-1,'integer width domain')
    require(gc['source_gsp_sha256']==proof['canonical_gsp_sha256'],'range source binding')
    require(proof['unique_cap32_query_terms_verified'] is True,'query weight/uniqueness audit missing')
    oracle=ledger.get(proof['oracle_run'])['payload']
    require(oracle['status']=='EXECUTED' and oracle['runtime']['config']['mode']=='cpu-sequential','unverified oracle run')
    require(oracle['runtime']['behavior']['contract']==U64 and oracle['runtime']['config']['docs']==proof['docs'] and oracle['runtime']['config']['query_count']==proof['query_count'] and oracle['runtime']['config']['k']==proof['k'],'oracle domain differs')
    require(roles(oracle)['index']==proof['canonical_gsp_sha256'] and roles(oracle)['queries']==proof['queries_sha256'],'oracle input identity differs')
    require(any(x['sha256']==proof['oracle_sha256'] for x in oracle['artifacts']),'oracle output not preserved')
    if wide['runtime']['config']['mode']=='gpusparse-u64-canonical':
        require(bound<=wide['runtime']['config']['selection_score_limit']==2**31-1,'native canonical selector domain')
        require(wide['runtime']['config']['max_conservative_score_bound']==bound,'wide bound mismatch')
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
    payload=dict(original['payload']);payload.update(status='REVIEWABLE',contract_mismatches=[],observed_contract_differences=sorted(mismatch),domain_equivalence_certificate=certificate_id,original_blocked_comparison=original['id'],qualification='Exact integer sums and canonical output semantics coincide only on this verified finite input domain. Physical accumulator widths remain distinct. All implementation/timing/resource differences remain disclosed; no native float/tie-contract equivalence or general-input guarantee.')
    return ledger.event('comparison',payload,[left_id,right_id,original['id'],certificate_id])
