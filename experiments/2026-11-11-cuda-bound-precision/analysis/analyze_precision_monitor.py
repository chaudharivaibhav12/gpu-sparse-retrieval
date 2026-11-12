"""Summarize sampled GPU context/clocks; does not prove exclusive GPU use."""
import argparse,csv,datetime as dt,json,statistics
from pathlib import Path
def analyze(path):
 blocks=[];current=None
 for line in Path(path).read_text().splitlines():
  if line.startswith('UTC '):
   current=dict(time=line[4:],gpu={},compute=[]);dt.datetime.fromisoformat(current['time']);blocks.append(current);continue
  if current is None or not line.strip():continue
  r=next(csv.reader([line]));r=[s.strip() for s in r]
  if len(r)==7 and r[0].isdigit() and r[1].startswith('GPU-'):current['gpu'][int(r[0])]=r
  elif len(r)==3 and r[0].startswith('GPU-') and r[1].isdigit():current['compute'].append(r)
 assert blocks and all(0 in b['gpu'] for b in blocks),'Missing target GPU rows'
 uuids={b['gpu'][0][1] for b in blocks};assert len(uuids)==1;target=next(iter(uuids));multi=[];zero=0;pids=set();active_clocks=[]
 for b in blocks:
  own={int(r[1]) for r in b['compute'] if r[0]==target};pids.update(own)
  if len(own)>1:multi.append(dict(utc=b['time'],pids=sorted(own)))
  if not own:zero+=1
  elif b['gpu'][0][3].split()[0].isdigit():active_clocks.append(int(b['gpu'][0][3].split()[0]))
 times=[dt.datetime.fromisoformat(b['time']) for b in blocks];gaps=[(b-a).total_seconds() for a,b in zip(times,times[1:])];assert all(x>0 for x in gaps)
 return dict(samples=len(blocks),target_gpu_uuid=target,observed_target_compute_pids=sorted(pids),snapshots_with_multiple_target_compute_pids=len(multi),concurrent_snapshots=multi,snapshots_without_target_compute_pid=zero,sample_gap_median_seconds=statistics.median(gaps) if gaps else None,sample_gap_max_seconds=max(gaps) if gaps else None,active_context_sampled_sm_clock_mhz=dict(minimum=min(active_clocks),maximum=max(active_clocks)) if active_clocks else None,qualification='Full run sampling includes setup, runtime gate, warmup and measurement; query timestamps cannot isolate measured rounds. PID ownership is not identified. Multiple target contexts flag possible contention; absence at sampled instants does not prove exclusive use. Clock values are sampled, not controlled. OtherGPU contexts are excluded.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('monitor');p.add_argument('output');a=p.parse_args();result=analyze(a.monitor)
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
