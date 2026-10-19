"""Render tables and scientific figures from verified large-scale results."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from large_pages import analyze_trace
from subpasses import write_csv

ROOT=Path(__file__).resolve().parents[1]/'experiments'
MiB=1024**2


def read(p):return list(csv.DictReader(p.open()))


def main():
    out=ROOT/'results-large-pages-15gib'
    fixture=json.loads((out/'fixture.json').read_text())
    pages=read(out/'summary.csv');traces=[]
    for row in pages:
        page=int(row['page_target_bytes'])
        t=dict(page_target_bytes=page,**analyze_trace(ROOT/f'traces/large-page{page}.sqlite',fixture['pass_limit'],out))
        assert t['decoder_blocks_total']==int(row['data_pages']), 'Decoder grid/page count mismatch'
        assert t['output_chunks']==int(row['output_chunks']), 'Trace/output table count mismatch'
        traces.append(t)
    write_csv(out/'trace-summary.csv',traces)
    # Use the same CSV representation for table/figure rendering.
    traces=read(out/'trace-summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    labels=[f"{int(r['page_target_bytes'])//1024} KiB" if int(r['page_target_bytes'])<MiB else f"{int(r['page_target_bytes'])//MiB} MiB" for r in pages]
    x=np.arange(len(pages));median=np.array([float(r['read_ms_median'])/1000 for r in pages])
    axes[0].bar(x,median,color='#176575')
    axes[0].errorbar(x,median,yerr=[median-np.array([float(r['read_ms_min'])/1000 for r in pages]),np.array([float(r['read_ms_max'])/1000 for r in pages])-median],fmt='none',capsize=4,color='black')
    axes[0].set_ylabel('Unprofiled reader time (seconds)');axes[0].set_title('Three measured rounds; observed min–max')
    axes[1].bar(x-.18,[float(r['header_ms']) for r in traces],.36,label='Page headers',color='#b78428')
    axes[1].bar(x+.18,[float(r['decoder_ms']) for r in traces],.36,label='Value decoding',color='#7550a0')
    axes[1].set_ylabel('GPU kernel time (ms, logarithmic)');axes[1].set_yscale('log');axes[1].legend();axes[1].set_title('One separate profiled read; full-read totals')
    for ax in axes:ax.set_xticks(x,labels);ax.set_xlabel('Page target');ax.grid(axis='y',alpha=.2)
    fig.suptitle('15 GiB uncompressed integer Page sweep\n60 Row Groups · fixed 2 GiB pass hint · filesystem-cache residency not guaranteed')
    fig.savefig(out/'pages.png',dpi=180);fig.savefig(out/'pages.svg');plt.close(fig)
    lines=['# 15 GiB uncompressed Page-size sweep','',
      f"{fixture['rows']:,} rows, 60 Row Groups, one INT64 column. Expected sum: {fixture['expected_sum']:,}.",
      'All 15 baseline and five separately profiled reads passed row, column and sum checks. No dictionary, statistics, compression, predicates or GDS.',
      'Three baseline rounds and one separate trace after an excluded warm-up per layout. The fixed pass hint is 2 GiB; output hint disabled.',
      'Each file exceeds available host RAM. No warm-cache guarantee or cache flushing; file I/O remains part of the read.', '',
      '| Page target | Actual data Pages | Pages / chunk | Reader median (s) | Reader min–max (s) | Input passes / subpasses | Header kernels (ms) | Value decoding (ms) | Decoder grids | Reader kernels |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for r,t,l in zip(pages,traces,labels):
        assert r['page_target_bytes']==t['page_target_bytes']
        lines.append(f"| {l} | {int(r['data_pages']):,} | {r['pages_per_chunk']} | {float(r['read_ms_median'])/1000:.3f} | {float(r['read_ms_min'])/1000:.3f}–{float(r['read_ms_max'])/1000:.3f} | {t['input_passes']} / {t['subpasses']} | {float(t['header_ms']):.3f} | {float(t['decoder_ms']):.3f} | {t['decoder_grids']} | {t['kernel_count']} |")
    lines+=['','![15 GiB Page sweep](pages.png)','',
      'Reader time sums synchronized constructor, has_next and read_chunk calls; reduction is separate. Every returned table is released after its sum.',
      'GPU timing columns describe one profiled read per layout, not repeated profiled medians. They include only has_next/read_chunk phases and exclude reductions. Kernel busy time is event-interval union, not SM utilization.',
      'Timeline CSVs contain an exact first-output-batch excerpt; trace-summary.csv contains totals for the full read. The article labels the different scopes.',
      'The original 64 MiB study used four Row Groups and a whole-file read. This new study changes layout, API, and cache conditions along with scale; compare Page targets within a dataset.',
      '[Method and commands](../LARGE_SCALE.md) · [Baseline summary](summary.csv) · [Trace summary](trace-summary.csv) · [Fixture](fixture.json)',
      'Full newly generated Nsight/SQLite captures remain local under experiments/traces/large-page*. The small CSV summaries and exact first-batch excerpts are versioned; the large captures follow the repository ignore policy.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    memory=read(ROOT/'results-large-15gib/memory-summary.csv')
    fig,ax=plt.subplots(figsize=(7.5,5),constrained_layout=True)
    for r in memory:
        peak=float(r['rmm_peak_bytes_median'])/MiB;seconds=float(r['read_ms_median'])/1000
        ax.scatter(peak,seconds,s=60,color='#176575');ax.annotate(f"{r['pass_limit_mib']} MiB",(peak,seconds),xytext=(6,6),textcoords='offset points')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Peak live RMM allocations (MiB)');ax.set_ylabel('Baseline reader time (seconds)');ax.grid(alpha=.25);ax.margins(.3)
    ax.set_title('15 GiB compressed integers: memory / latency\nThree timing rounds; two separate memory rounds')
    fig.savefig(ROOT/'results-large-15gib/large-memory.png',dpi=180);fig.savefig(ROOT/'results-large-15gib/large-memory.svg');plt.close(fig)
    print('Large-scale reports and figures generated')


if __name__=='__main__':main()
