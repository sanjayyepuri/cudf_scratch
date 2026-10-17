"""Capture real CUDA/NVTX events for the generated page sweep and summarize them."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess


def merged_duration(intervals):
    """Union duration: overlapping GPU events must not be counted twice."""
    total, end = 0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--nsys', type=Path, default=Path('/tmp/cudf-profiling/nsight/opt/nvidia/nsight-systems/2026.3.2/bin/nsys'))
    parser.add_argument('--manifest', type=Path, default=root/'experiments/results-pages/manifest.json')
    parser.add_argument('--output', type=Path, default=root/'experiments/results-pages-traced')
    parser.add_argument('--trace', type=Path, default=root/'experiments/traces/pages')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--analyze-only', action='store_true', help='Reuse the existing capture and SQLite export')
    parser.add_argument('--binary', type=Path, default=root/'bazel-bin/layout_benchmark')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    baseline = json.loads(args.manifest.read_text())
    layouts = baseline['layouts']
    if len({x['row_group_rows'] for x in layouts}) != 1:
        parser.error('input manifest must be a page-only sweep')
    args.output.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KVIKIO_COMPAT_MODE='ON', CUDF_PROFILE='1',
               NSYS_NVTX_PROFILER_REGISTER_ONLY='0')
    command = [str(args.nsys), 'profile', '--trace=cuda-sw,nvtx', '--sample=none',
               '--cpuctxsw=none', '--capture-range=nvtx', '--nvtx-capture=measured',
               '--capture-range-end=stop', '--force-overwrite=true',
               '--output='+str(args.trace), str(args.binary.absolute()), str(args.repeats),
               *[x['file'] for x in layouts]]
    if args.analyze_only:
        captured = subprocess.CompletedProcess(command, 0, stdout=(args.output/'capture.log').read_text(), stderr='')
    else:
        captured = subprocess.run(command, env=env, text=True, capture_output=True)
        (args.output/'capture.log').write_text(captured.stdout + captured.stderr)
        captured.check_returncode()
    # nsys may interleave diagnostics with the target's stdout; retain only validated CSV records.
    samples = []
    for line in captured.stdout.splitlines():
        row = next(csv.reader([line]))
        if len(row) == 6 and row[0] in {x['file'] for x in layouts}:
            sample = dict(zip(('file', 'iteration', 'rows', 'sum', 'read_ms', 'sum_ms'), row))
            if int(sample['rows']) != baseline['rows'] or int(sample['sum']) != baseline['expected_sum']:
                raise RuntimeError('Incorrect result in profiled run')
            samples.append(sample)
    if len(samples) != len(layouts) * args.repeats:
        raise RuntimeError('Missing profiled correctness samples')
    write_csv(args.output/'samples.csv', samples)
    database = args.trace.with_suffix('.sqlite')
    if not args.analyze_only:
        exported = subprocess.run([str(args.nsys), 'export', '--type=sqlite',
                                   '--force-overwrite=true', '--output='+str(database),
                                   str(args.trace.with_suffix('.nsys-rep'))], text=True, capture_output=True)
        (args.output/'export.log').write_text(exported.stdout + exported.stderr)
        exported.check_returncode()
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute('SELECT id, value FROM StringIds'))
    ranges = [dict(x) for x in connection.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL')]
    for event in ranges:
        event['label'] = event['text'] or strings.get(event['textId'], '')
    kernels = [dict(x) for x in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    copies = [dict(x) for x in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY')]
    if not kernels or not copies:
        raise RuntimeError('Capture lacks actual CUDA kernels or memory transfers')
    apis = [dict(x) for x in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    read_samples, kernel_rows, api_rows, timeline = [], [], [], []
    for outer in ranges:
        if not outer['label'].startswith('sample:'):
            continue
        file, iteration = outer['label'][len('sample:'):].rsplit(':', 1)
        layout = next(x for x in layouts if x['file'] == file)
        phases = [x for x in ranges if x['label'] == 'benchmark/read'
                  and x['globalTid'] == outer['globalTid']
                  and outer['start'] <= x['start'] < x['end'] <= outer['end']]
        if len(phases) != 1:
            raise RuntimeError('Expected one read range per measured sample')
        phase = phases[0]
        start, end = phase['start'], phase['end']
        active_kernels = [x for x in kernels if x['end'] > start and x['start'] < end]
        active_copies = [x for x in copies if x['end'] > start and x['start'] < end]
        def intervals(events):
            return [(max(start, x['start']), min(end, x['end'])) for x in events]
        kernel_time = merged_duration(intervals(active_kernels))
        copy_time = merged_duration(intervals(active_copies))
        combined_time = merged_duration(intervals(active_kernels + active_copies))
        running = peak = 0
        for _, delta in sorted([(a, 1) for a,b in intervals(active_kernels)] +
                               [(b, -1) for a,b in intervals(active_kernels)]):
            running += delta
            peak = max(peak, running)
        read_samples.append(dict(file=file, page_target_bytes=layout['page_target_bytes'],
                                 iteration=int(iteration), read_wall_ms=(end-start)/1e6,
                                 kernel_count=len(active_kernels), max_concurrent_kernels=peak, memcpy_count=len(active_copies),
                                 kernel_busy_ms=kernel_time/1e6, memcpy_busy_ms=copy_time/1e6,
                                 kernel_or_memcpy_ms=combined_time/1e6,
                                 traced_activity_percent=100*combined_time/(end-start),
                                 no_kernel_or_memcpy_ms=((end-start)-combined_time)/1e6,
                                 memcpy_bytes=sum(x['bytes'] for x in active_copies)))
        for x in active_kernels:
            name = strings[x['demangledName']]
            kernel_rows.append(dict(file=file, iteration=int(iteration), name=name,
                                    duration_us=(x['end']-x['start'])/1000,
                                    start_ns=x['start'], end_ns=x['end'], stream_id=x['streamId'],
                                    device_id=x['deviceId'], context_id=x['contextId'],
                                    grid_blocks=x['gridX']*x['gridY']*x['gridZ'],
                                    threads_per_block=x['blockX']*x['blockY']*x['blockZ'],
                                    registers_per_thread=x['registersPerThread'],
                                    dynamic_shared_bytes=x['dynamicSharedMemory']))
        for x in apis:
            if x['globalTid'] == phase['globalTid'] and x['end'] > start and x['start'] < end:
                api_rows.append(dict(file=file, iteration=int(iteration), name=strings[x['nameId']],
                                     duration_us=(min(end,x['end'])-max(start,x['start']))/1000))
        if int(iteration) == 1:
            for kind, events in [('kernel', active_kernels), ('memcpy', active_copies)]:
                for x in events:
                    timeline.append(dict(page_target_bytes=layout['page_target_bytes'],
                                         kind=kind, start_ms=(max(start,x['start'])-start)/1e6,
                                         duration_ms=(min(end,x['end'])-max(start,x['start']))/1e6))
    if len(read_samples) != len(layouts) * args.repeats:
        raise RuntimeError('Missing NVTX read ranges in trace')
    write_csv(args.output/'read-phases.csv', read_samples)
    write_csv(args.output/'kernels.csv', kernel_rows)
    write_csv(args.output/'timeline.csv', timeline)
    if api_rows:
        write_csv(args.output/'cuda-apis.csv', api_rows)
    summaries = []
    for layout in layouts:
        phases = [x for x in read_samples if x['file'] == layout['file']]
        decoder = [x for x in kernel_rows if x['file'] == layout['file'] and 'decode_page_data' in x['name']]
        headers = [x for x in kernel_rows if x['file'] == layout['file'] and
                   ('count_page_headers_kernel' in x['name'] or 'decode_page_headers_kernel' in x['name'])]
        summaries.append(dict(page_target_bytes=layout['page_target_bytes'], data_pages=layout['data_pages'],
                              baseline_read_ms=layout['read_ms_median'],
                              traced_read_ms=statistics.median(x['read_wall_ms'] for x in phases),
                              kernel_busy_ms=statistics.median(x['kernel_busy_ms'] for x in phases),
                              memcpy_busy_ms=statistics.median(x['memcpy_busy_ms'] for x in phases),
                              traced_activity_percent=statistics.median(x['traced_activity_percent'] for x in phases),
                              gap_ms=statistics.median(x['no_kernel_or_memcpy_ms'] for x in phases),
                              kernels_per_read=statistics.median(x['kernel_count'] for x in phases),
                              max_concurrent_kernels=max(x['max_concurrent_kernels'] for x in phases),
                              decoder_launches_per_read=len(decoder)/args.repeats,
                              header_ms=statistics.median([sum(x['duration_us'] for x in headers if x['iteration']==i)/1000
                                                         for i in range(1,args.repeats+1)]),
                              decoder_grid_blocks=','.join(map(str,sorted({x['grid_blocks'] for x in decoder}))),
                              decoder_ms=statistics.median([sum(x['duration_us'] for x in decoder if x['iteration']==i)/1000
                                                           for i in range(1,args.repeats+1)])))
    write_csv(args.output/'summary.csv', summaries)
    metadata = dict(nsys_version=subprocess.check_output([str(args.nsys),'--version'],text=True).strip(),
                    command=command, repeats=args.repeats, baseline_manifest=str(args.manifest),
                    trace_file=str(args.trace.with_suffix('.nsys-rep')), sqlite_file=str(database),
                    verified_samples=len(samples), kernel_events=len(kernels), memcpy_events=len(copies),
                    hardware_counters='Not requested; deferred by user. Counter probe reported insufficient privilege.')
    (args.output/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines = ['# Page-size sweep: CUDA trace', '',
             f'{len(samples)} profiled sums verified; {len(kernels)} kernel events and {len(copies)} memcpy events recorded.',
             f'{args.repeats} profiled rounds, following one excluded warm-up per layout. Compare performance using the separate unprofiled baseline.', '',
             '| Page target (KiB) | Pages | Baseline read (ms) | Traced read (ms) | Kernel busy (ms) | Memcpy busy (ms) | Activity (%) | Gap (ms) | Decoder blocks | Decoder time (ms) | Header kernels (ms) |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in summaries:
        lines.append(f"| {x['page_target_bytes']//1024:,} | {x['data_pages']} | {x['baseline_read_ms']:.3f} | {x['traced_read_ms']:.3f} | {x['kernel_busy_ms']:.3f} | {x['memcpy_busy_ms']:.3f} | {x['traced_activity_percent']:.1f} | {x['gap_ms']:.3f} | {x['decoder_grid_blocks']} | {x['decoder_ms']:.3f} | {x['header_ms']:.3f} |")
    lines += ['', 'All displayed times are per-layout medians. Columns need not add up: events can overlap and medians are taken independently.',
              'Activity means union time of recorded kernels and memcpy events divided by the enclosing synchronized read range.',
              'It is not SM utilization, occupancy, or a hardware counter. Gap means no recorded kernel or memcpy; it can include CPU work, file access, allocation, scheduling and profiling overhead.',
              'CUDA memsets, if any, are not included in this activity definition. Decoder time sums kernels whose names contain decode_page_data.',
              'Capture uses software CUDA tracing, NVTX labels and no CPU sampling or hardware counters.',
              'The first measured iteration supplies timeline.csv; use the full .nsys-rep in Nsight Systems for interactive inspection.']
    lines += ['', '| Page target (KiB) | Kernels/read | Data-decoder launches/read | Maximum concurrent kernels |',
              '|---:|---:|---:|---:|']
    for x in summaries:
        lines.append(f"| {x['page_target_bytes']//1024:,} | {x['kernels_per_read']:.0f} | {x['decoder_launches_per_read']:.0f} | {x['max_concurrent_kernels']} |")
    lines += ['', 'Concurrency is measured by overlap of kernel execution intervals, not by stream count or launch count.',
              'Grid size describes scheduled blocks, not simultaneously resident blocks.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
