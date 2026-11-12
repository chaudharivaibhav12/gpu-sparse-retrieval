from precision_profile_sequence import expected_launches
"""Checked per-kernel DRAM counters; excludes copies/memsets and replay timings."""
import argparse,collections,csv,json,re
from decimal import Decimal
from pathlib import Path
from analyze_pruning_trace import rows,selection_grids

def shape(text):
    m=re.fullmatch(r'\((\d+),\s*(\d+),\s*(\d+)\)',text);assert m,text
    return tuple(map(int,m.groups()))

def number(text):
    v=Decimal(text.replace(',',''));assert v.is_finite() and v>=0 and v==v.to_integral_value(),text
    return int(v)

def analyze(counters,native_csv,B,mode,precision):
    lines=Path(counters).read_text().splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID",'))
    table=list(csv.DictReader(lines[start:]));units=table.pop(0);assert units['ID']==''
    assert units['dram__bytes_read.sum']=='byte' and units['dram__bytes_write.sum']=='byte'
    assert all(r['ID'].isdigit() for r in table);assert [int(r['ID']) for r in table]==list(range(len(table)))
    assert len({r['Process ID'] for r in table})==1 and {r['Device'] for r in table}=={'0'}
    assert all(number(r['profiler__replayer_passes'])==1 and number(r['profiler__replayer_passes_type_warmup'])==0 for r in table),'Multiple counter replay/warmup passes need separate qualification'
    samples=rows(native_csv);assert len(samples)==25
    assert all(r['phase']=='measured' and int(r['round'])==0 and int(r['pruned'])==mode for r in samples)
    index=0;per=[];total_read=collections.Counter();total_write=collections.Counter()
    for sample in samples:
        expected=expected_launches(sample,B,mode,precision)
        read=collections.Counter();write=collections.Counter();first=index
        for stage,name,grid,threads in expected:
            assert index<len(table);r=table[index]
            assert re.search(r'\b'+name+r'\b',r['Kernel Name']),(index,name,r['Kernel Name'])
            assert shape(r['Grid Size'])==(grid,1,1) and shape(r['Block Size'])==(threads,1,1),(index,name,'launch dimensions')
            read[stage]+=number(r['dram__bytes_read.sum']);write[stage]+=number(r['dram__bytes_write.sum']);index+=1
        total_read.update(read);total_write.update(write)
        per.append(dict(query_id=sample['query_id'],kernel_launches=index-first,stage_kernel_dram_read_bytes=dict(read),stage_kernel_dram_write_bytes=dict(write),total_kernel_dram_read_bytes=sum(read.values()),total_kernel_dram_write_bytes=sum(write.values())))
    assert index==len(table),'Extra/unattributed counter row'
    return dict(block_size=B,mode=mode,precision=precision,queries=25,kernel_launches=index,stage_mean_per_query_kernel_dram_read_bytes={k:v/25 for k,v in total_read.items()},stage_mean_per_query_kernel_dram_write_bytes={k:v/25 for k,v in total_write.items()},mean_per_query_kernel_dram_read_bytes=sum(total_read.values())/25,mean_per_query_kernel_dram_write_bytes=sum(total_write.values())/25,per_query=per,replay_passes=1,replay_warmup_passes=0,cache_control='none',clock_control='none',qualification='Actual hardware DRAM counters summed only over checked profiled kernels. Excludes PCIe transfers, copy-engine memory traffic, score-reset memset, CPU work, setup and unprofiled operations. No ordinary latency, full-query traffic or peak-bandwidth claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('counters');p.add_argument('csv');p.add_argument('B',type=int,choices=[16,32]);p.add_argument('mode',type=int,choices=[0,1]);p.add_argument('precision',type=int,choices=[0,1,2]);p.add_argument('output');a=p.parse_args()
    result=analyze(a.counters,a.csv,a.B,a.mode,a.precision)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
