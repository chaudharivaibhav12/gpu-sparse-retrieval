"""Source-preservation/model checks; Rust execution is a separate build observation."""
import argparse,ast,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);a=p.parse_args();E=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
mat=load('materialize',E/'code/materialize.py');model=load('expected_cpu',E/'code/expected_cpu.py')
class Checks(unittest.TestCase):
 def test_python_syntax(self):
  for f in (E/'code').glob('*.py'):ast.parse(f.read_text())
 def test_parent_hashes_and_materialization_preserve_u64_and_serialization(self):
  with tempfile.TemporaryDirectory() as d:
   dest=Path(d)/'BMP';v=mat.materialize(a.parent,dest,E)
   original=(a.parent/'src/search.rs').read_text();self.assertEqual((dest/'src/search.rs').read_text().replace(mat.ACCESSORS,''),original)
   self.assertEqual((dest/'src/index/forward_index.rs').read_bytes(),(a.parent/'src/index/forward_index.rs').read_bytes())
   self.assertEqual((dest/'src/query/live_block.rs').read_bytes(),(a.parent/'src/query/live_block.rs').read_bytes())
   self.assertEqual((dest/'src/query/topk_heap.rs').read_bytes(),(a.parent/'src/query/topk_heap.rs').read_bytes())
   self.assertEqual((dest/'Cargo.lock').read_bytes(),(a.parent/'Cargo.lock').read_bytes())
   for name in json.loads((E/'configs/native-build-inputs.json').read_text())['files']:self.assertEqual((dest/name).read_bytes(),(E/'code/native-build'/name).read_bytes())
   with self.assertRaises(AssertionError):mat.materialize(a.parent,dest,E)
 def test_u32_search_has_only_declared_changes(self):
  original=(a.parent/'src/search.rs').read_text();start=original.index('fn search_with_ids(');end=original.index('\n#[cfg(test)]',start);body=original[start:end]
  body=body.replace('TopKHeap<u64>','TopKHeap<u32>').replace('p.kth(limit) as u64 * p.term_weight() as u64','p.kth(limit) as u32 * p.term_weight() as u32').replace('live_block::compute_upper_bounds_raw','compute_upper_bounds_raw').replace('live_block::compute_upper_bounds','compute_upper_bounds')
  self.assertIn(body,(E/'code/guarded_u32.rs').read_text())
 def test_arithmetic_bodies_have_only_width_and_visibility_delta(self):
  core=(E/'code/guarded_u32.rs').read_text();s=(a.parent/'src/index/forward_index.rs').read_text();body=s[s.index('#[inline]\npub fn block_score'):s.index('\n#[cfg(test)]')].replace('pub fn block_score','fn block_score').replace('u64','u32');self.assertIn(body,core)
  s=(a.parent/'src/query/live_block.rs').read_text();body=s[:s.index('\n#[cfg(test)]')].replace('pub fn compute_','fn compute_').replace('u64','u32');self.assertIn(body,core)
 def test_full_schema_and_width_only_cpu_delta(self):
  inputs=('/index','/queries',8841823,28679,1049282186,500,25,180540)
  wide=model.expected(*inputs,64,20,3);narrow=model.expected(*inputs,32,20,3)
  self.assertEqual(set(wide['config']),set(narrow['config']));self.assertEqual(set(wide['behavior']),set(narrow['behavior']))
  self.assertEqual({k for k in wide['config'] if wide['config'][k]!=narrow['config'][k]},{'score_bits','bound_score_bits','heap_score_bits'})
  self.assertEqual({k for k in wide['behavior'] if wide['behavior'][k]!=narrow['behavior'][k]},{'contract','arithmetic'})
  self.assertEqual(wide['config']['original_id_bits'],64)
 def test_model_rejects_unsupported_domain(self):
  args=['/i','/q',8841823,28679,1049282186,500,128,1044480,32,20,3]
  for index,value in [(8,16),(6,129),(7,1044481),(9,0),(10,-1)]:
   bad=list(args);bad[index]=value
   with self.assertRaises(AssertionError):model.expected(*bad)
 def test_same_adapter_timing_and_widening_body(self):
  for bits in [32,64]:
   s=(E/f'code/project_benchmark_u{bits}.rs').read_text();body=s[s.index('fn query('):s.index('fn monitor(')]
   self.assertLess(body.index('let start = Instant::now()'),body.index('let mut heaps'))
   self.assertLess(body.index('u64::from(r.score)'),body.index('start.elapsed()'))
   self.assertEqual(s.count('VerifiedCollection::new(&index, &forward, &context)?'),1)
   self.assertIn('gate(&a, &checked, &prepared)?',s)
   self.assertIn('type Score = u'+str(bits)+';',s)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','checks':result.testsRun,'qualification':'Python syntax,exact source preservation and expected-schema checks only;not Rust compilation or measurements'}));raise SystemExit(not result.wasSuccessful())
