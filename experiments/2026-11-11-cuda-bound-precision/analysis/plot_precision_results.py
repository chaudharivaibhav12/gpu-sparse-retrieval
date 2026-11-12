"""Plot completed, independently checked contrast JSONs; no benchmark compute."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot(inputs,destination):
 assert set(inputs)=={'cuda2','cuda5'}
 figures=[];fig,axes=plt.subplots(2,2,figsize=(9,6),sharey=True)
 upper=1.0
 for row,host in enumerate(['cuda2','cuda5']):
  data=json.loads(Path(inputs[host]).read_text());assert set(data)=={'16','32'}
  for col,B in enumerate(['16','32']):
   ax=axes[row,col];means=[1.0];intervals=[(1.0,1.0)];assert set(data[B])=={'1','2'}
   for precision in ['1','2']:
    s=data[B][precision];point=s['pruned_over_full'];low,high=s['crossed_query_process_bootstrap_95_interval']
    assert len(s['per_query'])==500 and len(s['process_ratios'])==3 and s['bootstrap_resamples']==10000
    assert all(math.isfinite(x) and x>0 for x in [point,low,high]) and low<=high
    means.append(point);intervals.append((low,high));upper=max(upper,point,high)
   ax.bar(range(3),means,color=['#788491','#437da8','#c18a42'],width=.58)
   for i,(point,(low,high)) in enumerate(zip(means,intervals)):
    if i:ax.vlines(i,low,high,color='black',lw=1.3);ax.hlines([low,high],i-.09,i+.09,color='black',lw=1.3)
    ax.text(i,max(point,high)+.025,f'{point:.3f}',ha='center',va='bottom',fontsize=9)
   ax.axhline(1,color='#444444',ls='--',lw=.9)
   ax.set_xticks(range(3),['Full byte','Coarse 4-bit','Selective'])
   ax.set_title(('RTX 2080 Ti' if host=='cuda2' else 'RTX 4070')+' · B='+B)
   ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
   if col==0:ax.set_ylabel('Query wall time / full byte')
 for ax in axes.flat:ax.set_ylim(0,upper*1.18)
 fig.suptitle('Bound precision: calibration latency',fontsize=14)
 fig.text(.5,.015,'500 queries · 3 processes per treatment · 20 measured rounds · conditional 95% bootstrap intervals\nFull-byte ratio is the reference. Shared nodes; no held-out or service-tail inference.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.07,1,.94]);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
 for suffix in ['svg','png']:
  p=destination/('precision-latency.'+suffix);fig.savefig(p,dpi=180);figures.append(str(p))
 plt.close(fig);return dict(figures=figures,matplotlib=matplotlib.__version__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cuda2');p.add_argument('cuda5');p.add_argument('destination');a=p.parse_args();print(json.dumps(plot(dict(cuda2=a.cuda2,cuda5=a.cuda5),a.destination)))
