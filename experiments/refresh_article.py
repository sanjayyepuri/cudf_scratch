"""Refresh the article's tables and observations from the new verified CSV data."""
import argparse
import csv
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
MiB=1024**2


def rows(path):return list(csv.DictReader(path.open()))
def table(headers,body):return '<Table data={'+json.dumps(dict(headers=headers,rows=body),ensure_ascii=False)+'} />'
def label(page):return f'{page//1024} KiB' if page<MiB else f'{page//MiB} MiB'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('blog',type=Path);args=parser.parse_args()
    data=json.loads((args.blog/'app/components/gpu-parquet/measurements.json').read_text())
    pages=data['largePages'];runs=data['largeRuns'];best=min(pages,key=lambda r:r['read'])
    page_dir=ROOT/'experiments/results-large-pages-15gib';mem_dir=ROOT/'experiments/results-large-15gib'
    traces=rows(page_dir/'trace-summary.csv');baseline=rows(mem_dir/'summary.csv')
    p=args.blog/'app/writing/posts/2026-10-17-gpu-parquet.mdx';s=p.read_text()
    replacements=[
      ('Exact page-size measurements',table(['Page target','Actual total pages','Median read','Observed min–max'],[[r['label'],f"{r['pages']:,}",f"{r['read']/1000:.3f} s",f"{r['low']/1000:.3f}–{r['high']/1000:.3f} s"] for r in pages])),
      ('Exact profiled kernel measurements',table(['Page target','Blocks per decoder launch','Header kernels','Value decoder','Kernel busy time','Memcpy busy time'],[[r['label'],r['grids'],f"{r['header']:.3f} ms",f"{r['decoder']:.3f} ms",f"{r['kernel']:.3f} ms",f"{r['memcpy']:.3f} ms"] for r in pages])),
      ('Exact pass-hint measurements',table(['Pass hint','Input passes / subpasses','Peak live RMM allocations','Baseline median read'],[[f"{r['passHint']:,} MiB",f"{r['passes']:,} / {r['subpasses']:,}",f"{r['memory']:,.2f} MiB",f"{r['read']/1000:.3f} s"] for r in runs]))]
    for heading,new in replacements:
        pattern=r'(<summary>'+re.escape(heading)+r'</summary>\s*)<Table[^\n]+/>'
        s,n=re.subn(pattern,lambda m:m[1]+new,s)
        if n!=1:raise RuntimeError(f'Could not replace {heading}')
    old="We can see that our results empirically agree with Luo's insight that each Column Chunk should have more pages (around 100). In my experiments, we saw that between 16 and 64 pages per Column Chunk provided the best performance. The paper mentions there is no one value that is _optimal_."
    new=f"In this larger sweep, **{best['label']} Pages** had the lowest median read time: **{best['read']/1000:.3f} s**, with **{best['perChunk']:,} Pages per Column Chunk**. This is a measured result for one layout and read configuration, not a universal optimum. File access, transfer, and batching are included, so the kernel trace helps distinguish GPU work from the overall read. The original small-scale results remain available in the figure."
    s=s.replace(old,new)
    a=s.index('My initial guess was that there was more overhead');b=s.index('### Memory Control',a)
    tiny,large=pages[0],pages[-1]
    kernel_counts=sorted({int(t['kernel_count']) for t in traces})
    s=s[:a]+f'''The new traces separate Page-header work from value decoding. At **{tiny['label']}**, header kernels took **{tiny['header']:.3f} ms** and value decoding took **{tiny['decoder']:.3f} ms**. At **{large['label']}**, those totals were **{large['header']:.3f} ms** and **{large['decoder']:.3f} ms**. These are totals from one separately profiled read per layout, not medians over repeated traces.

Header kernels walk Pages within Column Chunks; value-decoding kernels expose Page-level blocks. At the largest Page size, this larger fixture launched **{large['grids']} blocks per decoder launch**, across **{large['decoderLaunches']} decoder launches**. The original 64 MiB case had only four decoding blocks in one launch. The changed Row Group layout and chunked read path matter, so the original four-block explanation cannot simply be copied to the new dataset.

The new reads recorded **{', '.join(map(str,kernel_counts))} reader kernels**, depending on layout. These counts exclude reduction kernels. The former “33 kernels per read” observation applies only to the original whole-file experiment. Layout affects both work inside kernels and how the chunked reader batches that work. The timelines show recorded activity, not SM utilization, and gaps are not proof that the GPU is entirely idle.

''' + s[b:]
    a=s.index('The 16 MiB and 64 MiB hints both required');b=s.index('Peak live RMM allocations count',a)
    small=min(runs,key=lambda r:r['passHint']);medium=max(runs,key=lambda r:r['passHint']);fast=min(runs,key=lambda r:r['read'])
    moderate=next(r for r in runs if r['passHint']==2048)
    largest=max(runs,key=lambda r:r['passHint'])
    s=s[:a]+f'''The {small['passHint']:,} MiB hint used **{small['passes']:,} input passes and {small['subpasses']:,} subpasses**; the {medium['passHint']:,} MiB hint used **{medium['passes']:,} input passes and {medium['subpasses']:,} subpasses**. The smaller hint reduced peak live allocations from **{medium['memory']:,.2f} MiB to {small['memory']:,.2f} MiB**, while taking **{small['read']/medium['read']:.1f}×** as long to read the file. The fastest measured configuration was **{fast['passHint']:,} MiB**, at **{fast['read']/1000:.3f} s** and **{fast['memory']:,.2f} MiB** of peak live allocations.

A practical comparison is 2 GiB versus 8 GiB: reader time falls from **{moderate['read']/1000:.3f} s to {largest['read']/1000:.3f} s** (about **{100*(1-largest['read']/moderate['read']):.1f}%**), while peak tracked allocations grow from **{moderate['memory']/1024:.2f} GiB to {largest['memory']/1024:.2f} GiB**. Most of the speedup is already present at the more moderate budget on this fixture; the fastest measured setting need not be the best use of memory for a larger query.

The primary 256 MiB–8 GiB grid targets practical working-set sizes. A separate completed 16–1,024 MiB timing sweep is retained as a diagnostic: 16 MiB is smaller than one compressed Row Group, so it is primarily a batching stress case, not a recommended setting. Its interrupted trace and unmeasured memory peaks are not used here. Changes across the primary grid can affect both input-pass and subpass counts; timing differences cannot be attributed solely to subpass count.

The 15 GiB dataset is processed incrementally; it is not retained as one 15 GiB GPU output table. This is why a much smaller live working set can suffice for this sum. A query that retains the entire output or uses additional operators has different capacity requirements.

''' + s[b:]
    s=s.replace('The [full experiment report]', 'The [full experiment report]')
    s=s.replace('## Closing Out', '''The refresh includes **18 unprofiled timing reads, six profiled reads, and twelve memory-instrumented reads** for the compressed fixture, plus **15 unprofiled and five profiled reads** for the Page sweep. Every measured read passed the CPU-reference sum, row-count, and selected-column checks. The memory runs also ended with zero tracked live bytes. The [large-scale methodology and reproduction commands](https://github.com/sanjayyepuri/cudf_scratch/blob/BENCHMARK_REVISION/experiments/LARGE_SCALE.md) record the changed layouts, cache caveats, and omitted unlimited configuration.

## Closing Out''')
    p.write_text(s)
    print('Refreshed exact tables and trace-backed observations')


if __name__=='__main__':main()
