"""Data-independent scientific invariants; no heldout source/content access."""
import importlib.util,json,math,sys,tempfile
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'))
# Staged validation may provide the unchanged parent model directory explicitly.
if len(sys.argv)>1:sys.path.append(sys.argv[1])
import prepare_heldout_queries as p
import run_heldout as c
spec=importlib.util.spec_from_file_location('analysis_heldout',B/'analysis/analyze_heldout.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
checks=0
for text,vocab,wanted in [('x '*64+'y '*3,{'x':0,'y':1},[(0,32),(1,2)]),('oov '*100+'known '*3,{'known':7},[(7,1)]),('',{'x':0},[]),('x '*32,{'x':0},[(0,32)])]:
 assert p.transform(text,vocab)==wanted;checks+=1
try:p.transform(' '.join('t%d'%i for i in range(129)),{'t%d'%i:i for i in range(129)})
except ValueError:checks+=1
else:raise AssertionError('Unsupported129terms accepted')
template={'config':{'query_count':500,'max_query_terms':95,'max_conservative_score_bound':180540,'seed_ranges':64,'queries_path':'old'},'behavior':{'contract':'exact'}};good=dict(template['config'],query_count=6480,max_query_terms=128,max_conservative_score_bound=1044480,queries_path='new path');c.validate_method_state(good,template['behavior'],template);checks+=1
for delta in [{'seed_ranges':65},{'query_count':6479},{'max_query_terms':129},{'max_conservative_score_bound':2**24},{'unexpected':1}]:
 try:c.validate_method_state(dict(good,**delta),template['behavior'],template)
 except AssertionError:checks+=1
 else:raise AssertionError('Frozen drift accepted:'+str(delta))
try:c.validate_method_state(good,{'contract':'approximate'},template)
except AssertionError:checks+=1
else:raise AssertionError('Behavior drift accepted')
with tempfile.TemporaryDirectory() as d:
 f=Path(d)/'freeze.json';f.write_text(json.dumps({'kind':'decision','payload':{}}))
 try:p.prepare('never-read','never-read','never-read','never-read',Path(d)/'out',f)
 except (AssertionError,KeyError):checks+=1
 else:raise AssertionError('Preparation before method freeze accepted')
result=a.contrast([[2,4]]*3,[[1,2]]*3,draws=100);assert math.isclose(result['query_time_ratio_a_over_b'],2) and all(math.isclose(x,2) for x in result['conditional95']) and math.isclose(result['inverse_ratio'],.5);checks+=1
assert a.quantile([0,10],.025)==.25;checks+=1
print(json.dumps({'checks':checks,'status':'PASS','qualification':'Synthetic data and pure-state checks; no heldout contents or CUDA execution'}))
