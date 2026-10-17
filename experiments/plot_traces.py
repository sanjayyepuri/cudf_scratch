"""Render the recorded baseline, kernel timings and CUDA-event timelines."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

root = Path(__file__).resolve().parent
output = root/'results-pages-traced'
summary = list(csv.DictReader((output/'summary.csv').open()))
baseline = list(csv.DictReader((root/'results-pages/summary.csv').open()))
timeline = list(csv.DictReader((output/'timeline.csv').open()))
labels = ['16 KiB', '64 KiB', '256 KiB', '1 MiB', '16 MiB']
x = np.arange(len(summary))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(12, 10), layout='constrained')
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
a, b, c = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1]), fig.add_subplot(grid[1,:])
median = np.array([float(r['read_ms_median']) for r in baseline])
a.errorbar(x, median, yerr=[median-np.array([float(r['read_ms_min']) for r in baseline]),
                           np.array([float(r['read_ms_max']) for r in baseline])-median],
           fmt='o', capsize=5, color='#1d4ed8')
a.set(xticks=x, xticklabels=labels, ylabel='Read time (ms)', ylim=(0,32),
      title='A. Unprofiled reads: median and observed range')
a.grid(axis='y', alpha=.2)
b.bar(x-.18, [float(r['header_ms']) for r in summary], width=.36, color='#d97706', label='Count + decode page headers')
b.bar(x+.18, [float(r['decoder_ms']) for r in summary], width=.36, color='#7c3aed', label='Decode page data')
b.set(xticks=x, xticklabels=labels, ylabel='Kernel time (ms)',
      title='B. Profiled kernel timings: per-read medians')
b.legend(fontsize=9)
b.grid(axis='y', alpha=.2)
colors = {'kernel':'#7c3aed', 'memcpy':'#0891b2'}
for i, row in enumerate(summary):
    page = int(row['page_target_bytes'])
    for j, kind in enumerate(('kernel','memcpy')):
        intervals = [(float(t['start_ms']), float(t['duration_ms'])) for t in timeline
                     if int(t['page_target_bytes']) == page and t['kind'] == kind]
        c.broken_barh(intervals, (i*3+j, .75), facecolors=colors[kind])
c.set(yticks=[i*3+.85 for i in range(len(summary))], yticklabels=labels,
      xlabel='Milliseconds from read-range start', ylabel='Page target',
      title='C. Recorded CUDA activity: first measured read for each layout', xlim=(0,32))
c.invert_yaxis()
c.grid(axis='x',alpha=.2)
c.legend(handles=[Patch(color=colors[k],label=k.capitalize()) for k in colors], loc='lower right')
fig.suptitle('Parquet page size on RTX 3090 — fixed 2,097,152 rows/group', fontsize=15)
fig.supxlabel('Same 8,388,608 INT64 values • warm cache • uncompressed • no GDS\nKernel/transfer events show activity intervals, not SM utilization. Profiling changes timing.', fontsize=10)
fig.savefig(output/'tracing.png', dpi=160)
fig.savefig(output/'tracing.svg')
print(output/'tracing.png')
