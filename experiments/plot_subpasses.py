"""Plot the saved integer subpass benchmark; no GPU work is performed."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent/'results-subpasses'
rows = list(csv.DictReader((root/'summary.csv').open()))
pass_sweep = sorted([r for r in rows if int(r['chunk_limit_mib']) == 0], key=lambda r:int(r['pass_limit_mib']))
output_sweep = sorted([r for r in rows if int(r['pass_limit_mib']) == 64], key=lambda r:int(r['chunk_limit_mib']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')

def label(n): return 'Unlimited' if int(n)==0 else str(n)
def count(row,key):
    values = row[key].split(',')
    if len(values)!=1: raise RuntimeError('Trace counts varied; plot ranges instead of selecting one')
    return int(values[0])

def latency(ax,data,field,title):
    x = np.arange(len(data)); median=np.array([float(r['read_ms_median']) for r in data])
    low=np.array([float(r['read_ms_min']) for r in data]);high=np.array([float(r['read_ms_max']) for r in data])
    ax.errorbar(x,median,yerr=[median-low,high-median],fmt='o',capsize=4,color='#1d4ed8')
    ax.set(xticks=x,xticklabels=[label(r[field]) for r in data],ylabel='Reader time (ms, log scale)',yscale='log',title=title)
    ax.grid(axis='y',alpha=.25,which='both')
    for i,y in enumerate(median):ax.annotate(f'{y:,.0f}',(i,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.margins(y=.25)

latency(axes[0,0],pass_sweep,'pass_limit_mib','A. Pass hint sweep; output limit disabled')
axes[0,0].set_xlabel('Pass-memory hint (MiB)')
x=np.arange(len(pass_sweep));a=axes[0,1]
a.bar(x-.18,[count(r,'input_passes') for r in pass_sweep],width=.36,label='Input passes',color='#0891b2')
a.bar(x+.18,[count(r,'subpasses') for r in pass_sweep],width=.36,label='Page subpasses',color='#7c3aed')
a.set(xticks=x,xticklabels=[label(r['pass_limit_mib']) for r in pass_sweep],yscale='log',ylabel='Count (log scale)',xlabel='Pass-memory hint (MiB)',title='B. Actual counts from CUDA/NVTX trace')
a.legend();a.grid(axis='y',alpha=.2)
latency(axes[1,0],output_sweep,'chunk_limit_mib','C. Output hint sweep; pass hint fixed at 64 MiB')
axes[1,0].set_xlabel('Output-table hint (MiB)')
axes[1,0].set_yscale('linear')
axes[1,0].set_ylabel('Reader time (ms)')
axes[1,0].set_ylim(0,1450)
x=np.arange(len(output_sweep));a=axes[1,1]
a.bar(x-.18,[count(r,'subpasses') for r in output_sweep],width=.36,label='Page subpasses',color='#7c3aed')
a.bar(x+.18,[count(r,'traced_output_chunks') for r in output_sweep],width=.36,label='Output tables',color='#d97706')
a.set(xticks=x,xticklabels=[label(r['chunk_limit_mib']) for r in output_sweep],ylabel='Count',xlabel='Output-table hint (MiB)',title='D. Subpasses and output tables are different')
a.legend();a.grid(axis='y',alpha=.2)
fig.suptitle('Compressed integer Parquet: 8 row groups, 3 columns, RTX 3090',fontsize=15)
fig.supxlabel('33,554,432 rows • ZSTD • 1 MiB pages • warm cache • no GDS\nLatency: five unprofiled rounds, median and observed range. Counts: two separate traced rounds.',fontsize=10)
fig.savefig(root/'subpasses.png',dpi=160)
fig.savefig(root/'subpasses.svg')
print(root/'subpasses.png')
