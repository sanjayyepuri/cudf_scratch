"""15 GiB page sweep using bounded C++ reads rather than a whole-table output."""
import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess

import fastparquet
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from subpasses import trace_counts, write_csv
from trace_pages import merged_duration

ROOT = Path(__file__).resolve().parents[1]
GiB = 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/tmp/cudf-large-pages'))
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/results-large-pages-15gib')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reuse-fixture', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1: parser.error('repeats must be positive')
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    group_rows, groups, pass_limit = 33_554_432, 60, 2048*1024**2
    fixture = dict(rows=group_rows*groups, row_groups=groups, group_rows=group_rows,
                   decoded_bytes=15*GiB, columns=1, pass_limit=pass_limit, output_limit=0,
                   seed=42, codec='NONE', dictionary=False, page_version='1.0',
                   repeats=args.repeats, warmups=1,
                   cache_policy='No cache flush or residency guarantee; 15 GiB files exceed available host RAM. Excluded warm-up is not a warm-cache claim.')
    layouts=[]
    for page in (16_384,65_536,262_144,1_048_576,16_777_216):
        file=args.data_dir/f'page{page}.parquet'
        expected=0
        if not args.reuse_fixture:
            rng=np.random.default_rng(42)
            schema=pa.schema([('amount',pa.int64())])
            with pq.ParquetWriter(file,schema,compression='NONE',use_dictionary=False,
                    data_page_size=page,write_batch_size=1024,write_statistics=False,
                    column_encoding='PLAIN',data_page_version='1.0') as writer:
                for group in range(groups):
                    values=rng.integers(0,10000,size=group_rows,dtype=np.int64)
                    expected+=int(values.sum())
                    writer.write_table(pa.table({'amount':values}),row_group_size=group_rows)
                    if (group+1)%10==0:print(f'page={page} generated {group+1}/{groups} groups',flush=True)
        else:
            expected=json.loads((args.output/'fixture.json').read_text())['expected_sum']
        meta=fastparquet.ParquetFile(file)
        counts=[sum(s.count for s in g.columns[0].meta_data.encoding_stats if s.page_type in (0,3)) for g in meta.row_groups]
        if len(counts)!=groups:raise RuntimeError('Incorrect row group count')
        item=dict(file=str(file),page_target_bytes=page,data_pages=sum(counts),
                  pages_per_chunk=min(counts),pages_per_chunk_max=max(counts),file_bytes=file.stat().st_size)
        layouts.append(item)
        if 'expected_sum' in fixture and fixture['expected_sum']!=expected:raise RuntimeError('Different values across layouts')
        fixture['expected_sum']=expected
        fixture['layouts']=layouts
        (args.output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    env=dict(os.environ,KVIKIO_COMPAT_MODE='ON',CUDF_BENCH_SINGLE_COLUMN='1')
    env.pop('CUDF_PROFILE',None);env.pop('CUDF_MEMORY',None)
    binary=ROOT/'bazel-bin/subpass_benchmark'
    nsys=Path('/tmp/cudf-profiling/nsight/opt/nvidia/nsight-systems/2026.3.2/bin/nsys')
    summaries=[];traces=[]
    # Fixed shuffled configuration order; timing is separate from tracing.
    np.random.default_rng(42).shuffle(layouts)
    for layout in layouts:
        page=layout['page_target_bytes'];stem=args.output/f'page{page}'
        command=[str(binary),str(args.repeats),layout['file'],f'{pass_limit}:0']
        with stem.with_suffix('.csv').open('w') as raw,stem.with_suffix('.log').open('w') as log:
            subprocess.run(command,env=env,stdout=raw,stderr=log,check=True)
        samples=list(csv.DictReader(stem.with_suffix('.csv').open()))
        validate(samples,fixture,args.repeats)
        times=[float(r['read_ms']) for r in samples]
        summaries.append(dict(**layout,read_ms_median=statistics.median(times),read_ms_min=min(times),
                              read_ms_max=max(times),output_chunks=int(samples[0]['output_chunks']),
                              effective_gib_s=15/(statistics.median(times)/1000)))
        write_csv(args.output/'summary.csv',sorted(summaries,key=lambda r:r['page_target_bytes']))
        print(f'page={page} baseline median={statistics.median(times):.2f} ms, sums verified',flush=True)
    for layout in sorted(layouts,key=lambda r:r['page_target_bytes']):
        page=layout['page_target_bytes'];stem=args.output/f'trace{page}'
        trace=ROOT/f'experiments/traces/large-page{page}'
        trace.parent.mkdir(parents=True,exist_ok=True)
        command=[str(nsys),'profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none',
                 '--capture-range=nvtx','--nvtx-capture=measured','--capture-range-end=stop',
                 '--force-overwrite=true','--output='+str(trace),str(binary),'1',layout['file'],f'{pass_limit}:0']
        captured=subprocess.run(command,env=dict(env,CUDF_PROFILE='1',NSYS_NVTX_PROFILER_REGISTER_ONLY='0'),text=True,capture_output=True)
        stem.with_suffix('.log').write_text(captured.stdout+captured.stderr);captured.check_returncode()
        lines=captured.stdout.splitlines();header=next(x for x in lines if x.startswith('pass_limit,'))
        records=[dict(zip(header.split(','),x.split(','))) for x in lines if x.startswith(str(pass_limit)+',')]
        validate(records,fixture,1)
        subprocess.run([str(nsys),'export','--type=sqlite','--force-overwrite=true','--output='+str(trace.with_suffix('.sqlite')),str(trace.with_suffix('.nsys-rep'))],check=True,stdout=subprocess.DEVNULL)
        traces.append(dict(page_target_bytes=page,**analyze_trace(trace.with_suffix('.sqlite'),pass_limit,args.output)))
        write_csv(args.output/'trace-summary.csv',traces)
        print(f'page={page} traced, sum verified',flush=True)
    (args.output/'commands.json').write_text(json.dumps(dict(binary=str(binary),nsys=str(nsys),env=env,baseline_repeats=args.repeats,trace_repeats=1),indent=2)+'\n')


def validate(samples,fixture,repeats):
    if len(samples)!=repeats:raise RuntimeError('Missing samples')
    for r in samples:
        if int(r['rows'])!=fixture['rows'] or int(r['sum'])!=fixture['expected_sum'] or int(r['columns'])!=1:raise RuntimeError(f'Incorrect result: {r}')


def analyze_trace(path,pass_limit,output):
    counts=trace_counts(path,[(pass_limit,0)],1)[0]
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'))
    ranges=[dict(r) for r in db.execute('select * from NVTX_EVENTS where end is not null')]
    for r in ranges:r['label']=r['text'] or strings.get(r['textId'],'')
    sample=next(r for r in ranges if r['label'].startswith('sample:'))
    reads=[r for r in ranges if r['label'] in ('benchmark/has_next','benchmark/read_chunk') and sample['start']<=r['start']<r['end']<=sample['end']]
    kernels=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL') if any(p['start']<=r['start']<r['end']<=p['end'] for p in reads)]
    copies=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY') if any(p['start']<=r['start']<r['end']<=p['end'] for p in reads)]
    headers=[r for r in kernels if any(n in strings[r['demangledName']] for n in ('count_page_headers_kernel','decode_page_headers_kernel'))]
    decoders=[r for r in kernels if 'decode_page_data' in strings[r['demangledName']]]
    running=peak=0
    for _,delta in sorted([(r['start'],1) for r in kernels]+[(r['end'],-1) for r in kernels]):running+=delta;peak=max(peak,running)
    # Keep the full read's totals, but show an exact, labelled first-batch
    # excerpt rather than thousands of widened subpixel rectangles.
    first_read=min((p for p in reads if p['label']=='benchmark/read_chunk'),key=lambda p:p['start'])
    excerpt_start=min(p['start'] for p in reads)
    excerpt_end=first_read['end']
    events=[dict(kind=kind,start_ms=(r['start']-excerpt_start)/1e6,duration_ms=(r['end']-r['start'])/1e6)
            for kind,events in [('kernel',kernels),('memcpy',copies)] for r in events
            if excerpt_start<=r['start']<r['end']<=excerpt_end]
    write_csv(output/f'timeline-{path.stem.split("page")[-1]}.csv',events)
    return dict(input_passes=counts['input_passes'],subpasses=counts['subpasses'],output_chunks=counts['output_chunks'],
                header_ms=sum(r['end']-r['start'] for r in headers)/1e6,decoder_ms=sum(r['end']-r['start'] for r in decoders)/1e6,
                decoder_grids=','.join(map(str,sorted({r['gridX']*r['gridY']*r['gridZ'] for r in decoders}))),
                decoder_launches=len(decoders),decoder_blocks_total=sum(r['gridX']*r['gridY']*r['gridZ'] for r in decoders),kernel_count=len(kernels),max_concurrent_kernels=peak,
                kernel_busy_ms=merged_duration([(r['start'],r['end']) for r in kernels])/1e6,
                memcpy_busy_ms=merged_duration([(r['start'],r['end']) for r in copies])/1e6,
                timeline_end_ms=(excerpt_end-excerpt_start)/1e6)


if __name__=='__main__':main()
