"""Benchmark compressed multi-column, multi-row-group Parquet with explicit limits."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MiB = 1024**2
ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(path, group_rows, groups):
    rng = np.random.default_rng(42)
    expected, decoded_bytes = 0, 0
    writer = None
    try:
        for group in range(groups):
            amount = rng.integers(0, 100_000, size=group_rows, dtype=np.int64)
            customer = rng.integers(0, 100_000_000, size=group_rows, dtype=np.int64)
            quantity = rng.integers(1, 21, size=group_rows, dtype=np.int32)
            table = pa.table({'amount': amount, 'customer_id': customer, 'quantity': quantity})
            expected += int(amount.sum())
            decoded_bytes += group_rows * 20  # integer values only; validity masks add more
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression='zstd', compression_level=3,
                                           use_dictionary=False, data_page_size=MiB,
                                           write_statistics=True, write_page_index=False)
            writer.write_table(table, row_group_size=group_rows)
            print(f'Wrote row group {group+1}/{groups}', flush=True)
    finally:
        if writer is not None:
            writer.close()
    metadata = pq.read_metadata(path)
    assert metadata.num_row_groups == groups and metadata.num_rows == group_rows*groups
    return dict(file=str(path.resolve()), rows=group_rows*groups, row_group_rows=group_rows,
                row_groups=groups, expected_sum=expected, columns=3, seed=42,
                codec='ZSTD', compression_level=3, page_target_bytes=MiB,
                dictionary_columns=[], nulls=False,
                fixed_width_decoded_bytes=decoded_bytes, file_bytes=path.stat().st_size,
                pyarrow=pa.__version__, numpy=np.__version__, kernel=platform.release(),
                row_group_compressed_bytes=[sum(metadata.row_group(g).column(c).total_compressed_size
                                               for c in range(3)) for g in range(groups)])


def parse_samples(text, expected, repeats, configs):
    header = 'pass_limit,chunk_limit,iteration,rows,sum,columns,output_chunks,max_chunk_rows,read_ms,sum_ms,total_ms'
    samples = []
    for line in text.splitlines():
        cells = next(csv.reader([line]))
        if len(cells) == 11 and cells[0].isdigit() and cells[1].isdigit():
            row = dict(zip(header.split(','), cells))
            if int(row['rows']) != expected['rows'] or int(row['sum']) != expected['expected_sum'] or int(row['columns']) != expected['columns']:
                raise RuntimeError(f'Incorrect read result: {row}')
            samples.append(row)
    if len(samples) != repeats * len(configs):
        raise RuntimeError('Missing samples')
    return samples


def trace_counts(database, configs, repeats):
    db = sqlite3.connect(database); db.row_factory = sqlite3.Row
    strings = dict(db.execute('select id,value from StringIds'))
    events = [dict(r) for r in db.execute('select * from NVTX_EVENTS where end is not null')]
    for r in events:
        r['label'] = r['text'] or strings.get(r['textId'], '')
    kernels = [dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
    results = []
    for sample in events:
        if not sample['label'].startswith('sample:'):
            continue
        _, pass_limit, chunk_limit, iteration = sample['label'].split(':')
        inside = [r for r in events if sample['start'] <= r['start'] < r['end'] <= sample['end']]
        subpasses = sum(r['label'] == 'preprocess_subpass_pages' for r in inside)
        outputs = sum(r['label'] == 'read_chunk_internal' for r in inside)
        active = [r for r in kernels if sample['start'] <= r['start'] < r['end'] <= sample['end']]
        input_passes = sum('count_page_headers_kernel' in strings[r['demangledName']] for r in active)
        if subpasses < 1 or input_passes < 1:
            raise RuntimeError('Required subpass or input-pass trace evidence missing')
        results.append(dict(pass_limit=int(pass_limit),chunk_limit=int(chunk_limit),iteration=int(iteration),
                            input_passes=input_passes,subpasses=subpasses,output_chunks=outputs,
                            kernels=len(active)))
    if len(results) != repeats*len(configs):
        raise RuntimeError('Missing trace samples')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-rows',type=int,default=4_194_304)
    parser.add_argument('--groups',type=int,default=8)
    parser.add_argument('--pass-limits-mib',type=int,nargs='+',default=[0,16,64,256,512])
    parser.add_argument('--output-limits-mib',type=int,nargs='+',default=[0,8,32,128])
    parser.add_argument('--output-pass-mib',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--trace-repeats',type=int,default=2)
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--reuse-fixture',action='store_true')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=Path('/tmp/cudf-subpass-data'))
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/results-subpasses')
    parser.add_argument('--nsys',type=Path,default=Path('/tmp/cudf-profiling/nsight/opt/nvidia/nsight-systems/2026.3.2/bin/nsys'))
    args = parser.parse_args()
    if min(args.pass_limits_mib+args.output_limits_mib+[args.output_pass_mib])<0:
        parser.error('memory hints must be nonnegative')
    if min(args.group_rows,args.groups,args.repeats,args.trace_repeats)<1:
        parser.error('sizes and repetitions must be positive')
    args.data_dir.mkdir(parents=True,exist_ok=True); args.output.mkdir(parents=True,exist_ok=True)
    file = args.data_dir/'integers-zstd.parquet'
    fixture_info = args.data_dir/'integers-fixture.json'
    if args.reuse_fixture:
        fixture = json.loads(fixture_info.read_text())
        if fixture['row_groups'] != args.groups or fixture['row_group_rows'] != args.group_rows or not file.exists():
            raise RuntimeError('Fixture does not match requested layout')
    else:
        fixture = make_fixture(file,args.group_rows,args.groups)
        fixture_info.write_text(json.dumps(fixture,indent=2)+'\n')
    (args.output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    if args.generate_only:
        print(json.dumps(fixture,indent=2)); return
    # Vary one limit at a time; shared 64:0 case connects the two sweeps.
    configs = sorted({(p*MiB,0) for p in args.pass_limits_mib} |
                     {(args.output_pass_mib*MiB,c*MiB) for c in args.output_limits_mib})
    binary = ROOT/'bazel-bin/subpass_benchmark'
    limits = [f'{p}:{c}' for p,c in configs]
    env = dict(os.environ,KVIKIO_COMPAT_MODE='ON')
    env.pop('CUDF_PROFILE',None)
    env.pop('CUDF_MEMORY',None)
    env.pop('CUDF_BENCH_SINGLE_COLUMN',None)
    with (args.output/'baseline.log').open('w') as log, (args.output/'baseline-raw.csv').open('w') as raw:
        measured = subprocess.run([str(binary),str(args.repeats),str(file.resolve()),*limits],
                                  env=env,text=True,stdout=raw,stderr=log)
    measured.check_returncode()
    samples = parse_samples((args.output/'baseline-raw.csv').read_text(),fixture,args.repeats,configs)
    write_csv(args.output/'samples.csv',samples)
    print(f'{len(samples)} baseline sums verified',flush=True)
    if args.baseline_only:
        return
    trace = ROOT/'experiments/traces'/('subpasses' if args.output.name=='results-subpasses' else args.output.name);trace.parent.mkdir(parents=True,exist_ok=True)
    env.update(CUDF_PROFILE='1',NSYS_NVTX_PROFILER_REGISTER_ONLY='0')
    command = [str(args.nsys),'profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none',
               '--capture-range=nvtx','--nvtx-capture=measured','--capture-range-end=stop',
               '--force-overwrite=true','--output='+str(trace),str(binary),str(args.trace_repeats),str(file.resolve()),*limits]
    captured = subprocess.run(command,env=env,text=True,capture_output=True)
    (args.output/'capture.log').write_text(captured.stdout+captured.stderr)
    captured.check_returncode()
    profiled = parse_samples(captured.stdout,fixture,args.trace_repeats,configs)
    write_csv(args.output/'profiled-samples.csv',profiled)
    exported = subprocess.run([str(args.nsys),'export','--type=sqlite','--force-overwrite=true',
                               '--output='+str(trace.with_suffix('.sqlite')),str(trace.with_suffix('.nsys-rep'))],
                              text=True,capture_output=True)
    (args.output/'export.log').write_text(exported.stdout+exported.stderr);exported.check_returncode()
    counts = trace_counts(trace.with_suffix('.sqlite'),configs,args.trace_repeats)
    write_csv(args.output/'trace-counts.csv',counts)
    summary=[]
    for p,c in configs:
        group=[r for r in samples if int(r['pass_limit'])==p and int(r['chunk_limit'])==c]
        traced=[r for r in counts if r['pass_limit']==p and r['chunk_limit']==c]
        if any(r['output_chunks'] != int(next(s['output_chunks'] for s in profiled if int(s['pass_limit'])==p and int(s['chunk_limit'])==c and s['iteration']==str(r['iteration']))) for r in traced):
            raise RuntimeError('Output-chunk trace marker does not match reader return count')
        item=dict(pass_limit_mib=p//MiB,chunk_limit_mib=c//MiB)
        for field in ('read_ms','sum_ms','total_ms','output_chunks','max_chunk_rows'):
            item[field+'_median']=statistics.median(float(r[field]) for r in group)
        item.update(read_ms_min=min(float(r['read_ms']) for r in group),read_ms_max=max(float(r['read_ms']) for r in group),
                    input_passes=','.join(map(str,sorted({r['input_passes'] for r in traced}))),
                    subpasses=','.join(map(str,sorted({r['subpasses'] for r in traced}))),
                    traced_output_chunks=','.join(map(str,sorted({r['output_chunks'] for r in traced}))))
        summary.append(item)
    if not any(r['subpasses']>r['input_passes'] for r in counts):
        raise RuntimeError('No configuration exercised multiple subpasses per input pass')
    write_csv(args.output/'summary.csv',summary)
    metadata=dict(command=command,nsys=subprocess.check_output([str(args.nsys),'--version'],text=True).strip(),
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
                  baseline_repeats=args.repeats,trace_repeats=args.trace_repeats,verified_baseline=len(samples),
                  verified_profiled=len(profiled),subpass_marker='preprocess_subpass_pages',
                  input_pass_proxy='count_page_headers_kernel launches; one per input pass on this fixture without page indexes')
    (args.output/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Compressed multi-row-group subpass benchmark','',
           f"{fixture['rows']:,} rows; {fixture['row_groups']} row groups; three selected integer columns; ZSTD level 3; 1 MiB page target.",
           f"File size {fixture['file_bytes']/MiB:.1f} MiB; fixed-width decoded values alone {fixture['fixed_width_decoded_bytes']/MiB:.0f} MiB, plus validity masks.",
           f"All {len(samples)} baseline and {len(profiled)} profiled sums matched {fixture['expected_sum']:,}.",
           f"{args.repeats} baseline rounds and {args.trace_repeats} separate traced rounds; shuffled configs, one excluded warm-up per configuration; cache residency is not verified.",
           '','| Pass hint (MiB) | Output hint (MiB) | Input passes | Subpasses | Output chunks | Read median (ms) | Read min–max (ms) | Sum median (ms) | Total median (ms) |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['pass_limit_mib']} | {r['chunk_limit_mib']} | {r['input_passes']} | {r['subpasses']} | {r['traced_output_chunks']} | {r['read_ms_median']:.2f} | {r['read_ms_min']:.2f}–{r['read_ms_max']:.2f} | {r['sum_ms_median']:.2f} | {r['total_ms_median']:.2f} |")
    lines += ['', 'Zero means no limit. Both parameters are hints, not strict caps on GPU memory.',
              f'Pass sweep: output hint is 0 (unlimited). Output sweep: pass hint fixed at {args.output_pass_mib} MiB.',
              'Read time sums synchronized constructor, has_next and read_chunk calls. has_next can prepare/decompress data and must be timed.',
              'Sum time includes per-output-chunk GPU sum and scalar transfer. Total also includes buffer/reader destruction and loop overhead.',
              'Only amount is summed, but all three integer columns are decoded and every output table must have three columns.',
              'Input-pass counts use count_page_headers_kernel as a proxy, validated by this fixture having no page indexes. Subpass counts use preprocess_subpass_pages ranges.',
              'Output counts come from returned tables and are cross-checked against read_chunk_internal ranges in each traced sample.',
              'Fewer/smaller output tables are not evidence of more subpasses. Trace counts distinguish those boundaries.',
              'A pass limit below one compressed row-group size can still load that group and exceed the requested hint.',
              'Changing the pass hint can change input-pass and subpass counts together; do not attribute all differences solely to subpass count.',
              'The dataset is synthetic but includes multiple row groups, three flat integer columns with no null values or nesting.',
              'There are no filters or GDS, and no hardware utilization counters are collected.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__': main()
