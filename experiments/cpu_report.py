"""Join native CPU and GPU end-to-end timing baselines, preserving raw timings."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def read(path):return list(csv.DictReader(path.open()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--blog',type=Path);parser.add_argument('--revision');args=parser.parse_args()
    out=ROOT/'experiments/results-cpu-15gib'
    cpu=read(out/'summary.csv');method=json.loads((out/'method.json').read_text())
    compressed=next(r for r in read(ROOT/'experiments/results-large-15gib/summary.csv') if r['pass_limit_mib']=='2048')
    comparisons=[]
    for r in cpu:
        if r['workload']=='compressed':
            gpu_samples=[float(x['total_ms']) for x in read(ROOT/'experiments/results-large-15gib/samples.csv') if int(x['pass_limit'])==2048*1024**2 and int(x['chunk_limit'])==0]
            gpu=statistics.median(gpu_samples);label='ZSTD · three integers'
        else:
            page=int(r['workload'][4:]);gpu_samples=[float(x['total_ms']) for x in read(ROOT/f'experiments/results-large-pages-15gib/page{page}.csv')]
            gpu=statistics.median(gpu_samples)
            label=f'PLAIN · {page//1024} KiB Pages' if page<1024**2 else f'PLAIN · {page//1024**2} MiB Pages'
        comparisons.append(dict(workload=r['workload'],label=label,threads=int(r['threads']),
            cpu_read_ms=float(r['read_ms_median']),cpu_sum_ms=float(r['sum_ms_median']),
            cpu_total_ms=float(r['total_ms_median']),cpu_total_min_ms=float(r['total_ms_min']),cpu_total_max_ms=float(r['total_ms_max']),gpu_total_ms=gpu,gpu_total_min_ms=min(gpu_samples),gpu_total_max_ms=max(gpu_samples),gpu_over_cpu=gpu/float(r['total_ms_median']),
            cpu_process_peak_rss_bytes=int(r['process_peak_rss_bytes'])))
    with (out/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    table=dict(headers=['Workload','CPU workers','CPU total: median (range)','GPU total: median (range), 2 GiB hint','GPU / CPU time'],rows=[
        [r['label'],str(r['threads']),f"{r['cpu_total_ms']/1000:.3f} s ({r['cpu_total_min_ms']/1000:.3f}–{r['cpu_total_max_ms']/1000:.3f})",f"{r['gpu_total_ms']/1000:.3f} s ({r['gpu_total_min_ms']/1000:.3f}–{r['gpu_total_max_ms']/1000:.3f})",f"{r['gpu_over_cpu']:.2f}×"] for r in comparisons])
    lines=['# Native C++ CPU baseline on the same 15 GiB Parquet files','',
        f"CPU: {method['cpu_model']}; {method['available_logical_cpus']} available logical CPUs. Arrow C++ 19.0.1; same hermetic LLVM toolchain as the GPU driver.",
        f"All {method['verified_samples']} measured CPU reads passed row count, selected-column count and CPU-reference sum checks. Three rounds after one excluded warm-up per configuration.",
        '', '| Workload | CPU workers | CPU read (s) | CPU sum (s) | CPU total (s) | GPU total (s) | GPU / CPU time |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:lines.append(f"| {r['label']} | {r['threads']} | {r['cpu_read_ms']/1000:.3f} | {r['cpu_sum_ms']/1000:.3f} | {r['cpu_total_ms']/1000:.3f} | {r['gpu_total_ms']/1000:.3f} | {r['gpu_over_cpu']:.2f}× |")
    lines+=['', '**GPU / CPU time below 1 means the GPU driver was faster; above 1 means the CPU driver was faster.** GPU uses a fixed 2 GiB pass hint and disabled output hint, not the fastest hint selected after looking at results.',
        '', 'CPU reader: one independent Arrow Parquet reader per fixed worker. Each wave reads at most one row group per worker, reduces the amount columns in parallel, and releases all tables before the next wave. Nested Arrow threading and prebuffering are disabled. Both drivers decode the same selected columns and reduce only amount.',
        'The ZSTD fixture tests 1, 8 and 32 workers. Uncompressed Page fixtures use 8 workers to keep decoded waves at most 2 GiB before scratch and input buffers; 32 workers could hold 8 GiB of values plus encoded inputs and exceed host RAM. Eight workers are not a claim of full-machine CPU saturation or best CPU performance.',
        'Read and sum phases are timed separately, with total including metadata, worker setup and cleanup. GPU total includes CUDA allocation, synchronization, copies and per-table scalar transfers. CPU and GPU batching and library implementations differ; this is an explicit implementation baseline, not a GPU-only kernel speedup or universally optimized engine comparison.',
        'CPU and GPU runs do not overlap. Cache flushing/residency is not controlled. Compressed input is 5.9 GiB; uncompressed files are each 15 GiB and exceed available host RAM. File-cache state, I/O implementations, scheduling and batch policy can affect the comparison.',
        'Each timing column is an independently computed median; read and sum medians need not add exactly to total median. See raw samples and observed min–max ranges in summary.csv.',
        'CPU RSS is a process-lifetime high-water mark including excluded warm-up, native allocator caches and libraries. It is not directly comparable to peak live GPU RMM requests.',
        '', '```bash','bazelisk build //:cpu_benchmark','/tmp/cudf-layout-venv/bin/python experiments/cpu_baseline.py','/tmp/cudf-layout-venv/bin/python experiments/cpu_report.py','```','',
        '[Raw samples](samples.csv) · [CPU summary](summary.csv) · [Joined comparison](comparison.csv) · [Hardware and method](method.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if args.blog:
        path=args.blog/'app/writing/posts/2026-10-17-gpu-parquet.mdx';s=path.read_text()
        section='''## CPU Baseline

> **Suggested analysis — CPU baseline.** The CPU baseline uses a native C++ Arrow 19.0.1 reader on the exact same files and columns, followed by an INT64 sum of amount. It uses the same hermetic LLVM toolchain and does not link CUDA or Python. The compressed fixture tests 1, 8 and 32 row-group workers. The uncompressed fixtures use 8 workers to keep decoded waves at most 2 GiB before input buffers and scratch; this is a bounded baseline, not a claim of fully optimized CPU throughput.

> The table compares **end-to-end read plus sum plus cleanup**, using a fixed **2 GiB GPU pass hint** for every workload. This differs from the reader-only times above. Every CPU configuration has three measured rounds after an excluded warm-up; all '''+str(method['verified_samples'])+''' measured sums and row counts passed. CPU and GPU runs do not overlap.

<Table data={'''+json.dumps(table,ensure_ascii=False)+'''} />

> **GPU / CPU time below 1 means the GPU driver was faster; above 1 means the CPU driver was faster.** These are different reader implementations and batching policies. GPU timing includes host-to-device transfers, explicit synchronization and per-table scalar transfers. Host file-cache residency is not controlled, and a 15 GiB uncompressed file exceeds available host RAM. The comparison is a baseline for these drivers and settings, not an isolated decoder-kernel speedup. Parentheses show observed min–max ranges, not confidence intervals. The 32-worker CPU and 2 GiB GPU totals on the compressed file are comparable within the observed CPU range; the GPU is not uniformly faster across the Page cases.

[CPU source, raw results and method](https://github.com/sanjayyepuri/cudf_scratch/blob/CPU_BENCHMARK_REVISION/experiments/results-cpu-15gib/report.md).

'''
        if '## CPU Baseline' in s:s=s[:s.index('## CPU Baseline')]+section+s[s.index('## Closing Out'):]
        else:s=s.replace('## Closing Out',section+'## Closing Out')
        s=s.replace('Layout affects both work inside kernels and how the chunked reader batches that work.', 'All five large layouts used 30 input passes and 30 subpasses, with 30 value-decoder launches. Their different reader-kernel counts therefore reflect helper work rather than a change in batch count.')
        revision=args.revision or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        s=s.replace('CPU_BENCHMARK_REVISION',revision)
        path.write_text(s)
    print('\n'.join(lines))


if __name__=='__main__':main()
