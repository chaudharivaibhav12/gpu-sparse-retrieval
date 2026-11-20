"""Safety/schedule/model/math checks without research execution."""
import argparse,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path,PurePosixPath
E=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,default=E.parents[1]);args=parser.parse_args();sys.path.insert(0,str(E/'code'));import run_study as study
class Checks(unittest.TestCase):
 def test_fixed_schedule_has_three_rotations_and_no_duplicate_slot(self):
  slots=study.schedule();self.assertEqual(len(slots),9);self.assertEqual(len(set(slots)),9)
  for b in range(3):self.assertEqual([s for block,s in slots if block==b],study.ORDER[b:]+study.ORDER[:b])
 def test_approval_is_exact_source_output_and_scope_bound(self):
  v=dict(approved=True,experiment=study.E,source_commit='a'*40,root='/root',output='/root/out',scope='one-fixed-calibration-study',resource_attempts_per_stage=3,technical_resubmissions=0,queries=500,heldout_queries=0);study.approval(v,'a'*40,PurePosixPath('/root'),PurePosixPath('/root/out'))
  for key,value in [('approved',False),('experiment','other'),('source_commit','b'*40),('root','/else'),('output','/root/retry'),('scope','anything'),('resource_attempts_per_stage',4),('technical_resubmissions',1),('queries',6480),('heldout_queries',1)]:
   changed=dict(v);changed[key]=value
   with self.assertRaises(AssertionError):study.approval(changed,'a'*40,PurePosixPath('/root'),PurePosixPath('/root/out'))
 def test_protocol_rejects_semantic_resource_or_order_drift(self):
  p=json.loads((E/'configs/protocol.json').read_text());study.validate_protocol(p)
  for key,value in [('k',100),('batch',2),('queries',6480),('process_blocks',4),('cpu',1),('warmups',2)]:
   c=copy.deepcopy(p);c['scope'][key]=value
   with self.assertRaises(AssertionError):study.validate_protocol(c)
  c=copy.deepcopy(p);c['resources']['own_swap_gib_max']=2
  with self.assertRaises(AssertionError):study.validate_protocol(c)
  c=copy.deepcopy(p);c['timing_order'][0].reverse()
  with self.assertRaises(AssertionError):study.validate_protocol(c)
 def test_missing_malformed_runner_results_are_not_invented_finishes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'runner.json'
   for data in ['', '[]','{}','{"id":"run","payload":[]}']:
    p.write_text(data)
    with self.assertRaises(ValueError):study.runner_result(p)
   p.write_text('{"id":"run","payload":{"status":"EXECUTED"}}');self.assertEqual(study.runner_result(p)['id'],'run')
 def test_physical_cpu_audit_cannot_silently_omit_guard_or_change_width(self):
  v=dict(docs=8841823,vocab=28679,nnz=1049282186,block_size=32,score_bits=32,full_forward_unique=True,raw_bounds_conservative=True,query_count=500,max_query_terms=25,max_conservative_score_bound=180540)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'audit.json';p.write_text(json.dumps(v));study.cpu_audit(p,32,25,180540)
   for key,value in [('score_bits',64),('raw_bounds_conservative',False),('full_forward_unique',False),('query_count',499),('nnz',1)]:
    c=dict(v);c[key]=value;p.write_text(json.dumps(c))
    with self.assertRaises(AssertionError):study.cpu_audit(p,32,25,180540)
 def test_orphan_cleanup_requires_our_uid_session_and_output_identity(self):
  sent=[];send=lambda group,sig:sent.append((group,sig));valid=[dict(pid=99,uid=7,session=99,matches_output=True)]
  result=study.signal_own_remainder(99,Path('/run'),rows=valid,send=send,uid=7);self.assertEqual(result['decision'],'signaled-verified-owned-remainder');self.assertEqual(sent,[(99,study.signal.SIGTERM)])
  for key,value in [('uid',8),('session',100),('matches_output',False)]:
   changed=[dict(valid[0],**{key:value})];sent.clear();result=study.signal_own_remainder(99,Path('/run'),rows=changed,send=send,uid=7);self.assertEqual(result['decision'],'refused-unresolved-ownership');self.assertEqual(sent,[])
  self.assertEqual(study.signal_own_remainder(99,Path('/run'),rows=[],send=send,uid=7)['decision'],'no-remaining-members')
 def test_frozen_bootstrap_direction_and_null(self):
  parent=args.repository/'experiments/2026-11-18-matched-final-evaluation/analysis/analyze_calibration.py'
  spec=importlib.util.spec_from_file_location('stats',parent);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  result=m.contrast([[1.,2.]]*3,[[2.,4.]]*3,draws=50);self.assertAlmostEqual(result['query_time_ratio_a_over_b'],.5);self.assertAlmostEqual(result['inverse_ratio'],2.);self.assertEqual(result['inverse_conditional95'],[2.,2.])
  result=m.contrast([[1.,2.]]*3,[[1.,2.]]*3,draws=50);self.assertEqual(result['conditional95'],[1.,1.])
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));print(json.dumps({'status':'PASS' if r.wasSuccessful() else 'FAIL','checks':r.testsRun,'scope':'Pure schedule/approval/protocol/audit/runner-result/estimator checks;no CIMS or query execution'}));raise SystemExit(not r.wasSuccessful())
