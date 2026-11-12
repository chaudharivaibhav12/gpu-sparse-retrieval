from precision_profile_sequence import expected_launches
"""Attribute an unchanged native run's kernels by checked launch sequence."""
import argparse,collections,csv,json,re,sqlite3,statistics
from pathlib import Path

def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def selection_grids(count,k):
    if not count or not k:return []
    answer=[]
    while True:
        groups=(count+255)//256;answer.append(groups)
        if groups==1:return answer
        count=groups*k

def analyze(database,native_csv,B,mode,precision):
    samples=rows(native_csv);assert len(samples)==25
    assert all(r['phase']=='measured' and int(r['round'])==0 and int(r['pruned'])==mode for r in samples)
    c=sqlite3.connect('file:'+str(Path(database).resolve())+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    names=dict(c.execute('select id,value from StringIds'));kernels=list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
    assert len({r['globalPid'] for r in kernels})==1 and {r['deviceId'] for r in kernels}=={0}
    index=0;per=[];totals=collections.Counter()
    for sample in samples:
        expected=expected_launches(sample,B,mode,precision)
        stage_ns=collections.Counter();first=index
        for stage,name,grid,threads in expected:
            assert index<len(kernels),'Missing kernel launch'
            kernel=kernels[index];actual=names[kernel['shortName']]
            assert re.search(r'\b'+name+r'\b',actual),(index,name,actual)
            assert (kernel['gridX'],kernel['gridY'],kernel['gridZ'],kernel['blockX'],kernel['blockY'],kernel['blockZ'])==(grid,1,1,threads,1,1),(index,name,'launch dimensions')
            assert kernel['end']>=kernel['start']
            if index:assert kernel['start']>=kernels[index-1]['end'],'Unexpected overlapping/default-stream kernel'
            stage_ns[stage]+=kernel['end']-kernel['start'];index+=1
        totals.update(stage_ns);per.append(dict(query_id=sample['query_id'],kernel_launches=index-first,stage_gpu_ms={k:v/1e6 for k,v in stage_ns.items()},total_kernel_gpu_ms=sum(stage_ns.values())/1e6))
    assert index==len(kernels),'Extra/unattributed kernel launch'
    kinds={r['id']:r['name'] for r in c.execute('select * from ENUM_CUDA_MEM_KIND')}
    operations={r['id']:r['name'] for r in c.execute('select * from ENUM_CUDA_DEV_MEM_EVENT_OPER')}
    active={};current=peak=0;allocation_events=0
    for r in c.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS order by start'):
        assert kinds[r['memKind']]=='CUDA_MEMOPR_MEMORY_KIND_DEVICE','Unaccounted memory kind'
        key=(r['deviceId'],r['contextId'],r['address']);op=operations[r['memoryOperationType']]
        if op=='CUDA_DEV_MEM_EVENT_OPR_ALLOCATION':
            assert key not in active;active[key]=r['bytes'];current+=r['bytes'];allocation_events+=1;peak=max(peak,current)
        elif op=='CUDA_DEV_MEM_EVENT_OPR_DEALLOCATION':
            assert active.pop(key)==r['bytes'];current-=r['bytes'];assert current>=0
        else:raise AssertionError('Unaccounted allocation operation '+op)
    assert not active and current==0,'Trace missing allocation/free lifetime'
    declared={int(r['device_allocation_bytes']) for r in samples};assert declared=={peak},('Tracked/declared buffer accounting differs',declared,peak)
    c.close()
    return dict(block_size=B,mode=mode,precision=precision,queries=25,kernel_launches=index,stage_total_gpu_ms={k:v/1e6 for k,v in totals.items()},stage_mean_per_query_gpu_ms={k:v/25e6 for k,v in totals.items()},tracked_application_peak_device_bytes=peak,allocation_events=allocation_events,per_query=per,qualification='Nsight diagnostic kernel durations only; excludes transfers, memset, CPU work and stream gaps. No ordinary latency or board/context peak claim. Stage labels require every kernel name, launch geometry and query count to match the native sequence.',memory_qualification='Sum of tracked application allocation lifetimes, validated against exact native buffer accounting; excludes CUDA context/driver/profiler overhead.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('database');p.add_argument('csv');p.add_argument('B',type=int,choices=[16,32]);p.add_argument('mode',type=int,choices=[0,1]);p.add_argument('precision',type=int,choices=[0,1,2]);p.add_argument('output');a=p.parse_args()
    result=analyze(a.database,a.csv,a.B,a.mode,a.precision)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
