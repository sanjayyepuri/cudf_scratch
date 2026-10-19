"""Run the native Arrow CPU baseline separately from GPU work, on identical files."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def read(path):return list(csv.DictReader(path.open()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--compressed-threads',type=int,nargs='+',default=[1,8,32])
    parser.add_argument('--page-threads',type=int,nargs='+',default=[8])
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/results-cpu-15gib')
    args=parser.parse_args()
    available=len(os.sched_getaffinity(0))
    if args.repeats<1 or min(args.compressed_threads+args.page_threads)<1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    compressed=json.loads((ROOT/'experiments/results-large-15gib/fixture.json').read_text())
    pages=json.loads((ROOT/'experiments/results-large-pages-15gib/fixture.json').read_text())
    configs=[('compressed',compressed['file'],compressed,'three',t) for t in args.compressed_threads if t<=available]
    configs += [(f"page{p['page_target_bytes']}",p['file'],pages,'single',t) for p in pages['layouts'] for t in args.page_threads if t<=available]
    random.Random(42).shuffle(configs)
    samples=[]
    for name,file,fixture,mode,threads in configs:
        stem=args.output/f'{name}-threads{threads}'
        command=[str(ROOT/'bazel-bin/cpu_benchmark'),str(args.repeats),file,str(threads),mode]
        print(f'begin {name} threads={threads}',flush=True)
        with stem.with_suffix('.csv').open('w') as raw,stem.with_suffix('.log').open('w') as log:
            subprocess.run(command,stdout=raw,stderr=log,check=True)
        group=read(stem.with_suffix('.csv'))
        if len(group)!=args.repeats:raise RuntimeError('Missing CPU samples')
        for r in group:
            if int(r['rows'])!=fixture['rows'] or int(r['sum'])!=fixture['expected_sum'] or int(r['columns'])!=(1 if mode=='single' else 3):
                raise RuntimeError(f'CPU correctness failure: {r}')
            samples.append(dict(workload=name,**r))
        print(f"done {name} threads={threads} median_total_ms={statistics.median(float(r['total_ms']) for r in group):.3f}",flush=True)
    with (args.output/'samples.csv').open('w') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    summary=[]
    for name,_,_,_,threads in sorted(configs):
        group=[r for r in samples if r['workload']==name and int(r['threads'])==threads]
        row=dict(workload=name,threads=threads)
        for key in ('read_ms','sum_ms','total_ms','max_wave_value_bytes'):
            row[key+'_median']=statistics.median(float(r[key]) for r in group)
        row.update(total_ms_min=min(float(r['total_ms']) for r in group),total_ms_max=max(float(r['total_ms']) for r in group),
                   process_peak_rss_bytes=max(int(r['process_peak_rss_bytes']) for r in group))
        summary.append(row)
    with (args.output/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    (args.output/'method.json').write_text(json.dumps(dict(cpu_model=cpu_model,available_logical_cpus=available,
        kernel=platform.release(),arrow='19.0.1',repeats=args.repeats,excluded_warmups=1,verified_samples=len(samples),
        implementation='Native C++ Arrow Parquet reader and reductions. Independent reader per fixed worker; nested Arrow thread pools and prebuffering disabled.',
        batching='At most one row group per worker in a wave; parallel read then parallel sum, then release tables. Page workloads use eight workers to bound host RAM.',
        cache='No flush or residency guarantee. Exact same files as GPU; all CPU runs occur after GPU runs finish.',
        memory='process_peak_rss_bytes is the process lifetime high-water mark, including warm-up and allocator caches; not comparable to RMM live allocations.',
        comparison='Use total_ms (read+sum+cleanup), including worker setup. Different libraries and batching policies: a baseline, not an optimized best-CPU claim.'),indent=2)+'\n')
    print(f'{len(samples)} measured CPU sums verified',flush=True)


if __name__=='__main__':main()
