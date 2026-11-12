"""Expected launches from reviewed 07d4990 native precision source."""
from analyze_pruning_trace import rows,selection_grids
def expected_launches(sample,B,mode,precision):
 N=int(sample['docs']);k=int(sample['k']);blocks=(N+B-1)//B;groups=(blocks+255)//256;expected=[]
 assert int(sample['precision'])==precision and int(sample['pruned'])==mode
 def add(stage,name,grid,threads):expected.append((stage,name,grid,threads))
 def compact(stage):
  add(stage,'flag_scan',groups,256);add(stage,'group_offsets',1,1);add(stage,'flag_scatter',groups,256)
 if mode:
  seeds=int(sample['seed_blocks']);seed_docs=int(sample['seed_docs'])
  if seeds:add('seed_score','block_scores',seeds,32)
  for grid in selection_grids(seed_docs,k):add('seed_topk','local_topk',grid,256)
  if blocks:
   add('bounds','full_bounds' if precision==0 else 'coarse_bounds',groups,256)
   if precision==2:
    compact('uncertainty_compaction');refined=int(sample['refined_blocks'])
    assert 0<=refined<=blocks-seeds
    if refined:add('refinement','refine_bounds',(refined+255)//256,256)
   compact('compaction')
 alive=int(sample['active_scored_blocks'])
 if alive:add('survivor_score' if mode else 'control_score','block_scores',alive,32)
 for grid in selection_grids(N,k):add('global_topk','local_topk',grid,256)
 return expected
