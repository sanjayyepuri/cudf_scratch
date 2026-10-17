"""Generate identical data in configurable layouts; run the C++ GPU reader and summarize."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

import fastparquet
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=8_388_608)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--sweep', choices=('both', 'row-groups', 'pages'), default='both')
    parser.add_argument('--row-group-page-bytes', type=int, default=65_536)
    parser.add_argument('--row-group-sizes', type=int, nargs='+',
                        default=[131_072, 524_288, 2_097_152, 8_388_608])
    parser.add_argument('--data-dir', type=Path, default=Path('/tmp/cudf-layout-data'))
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1] / 'bazel-bin/layout_benchmark')
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error('rows and repeats must be positive')
    if args.row_group_page_bytes < 1 or any(size < 1 for size in args.row_group_sizes):
        parser.error('page target and row-group sizes must be positive')
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    # Uniform non-null INT64s, identical order and values in every file.
    values = np.random.default_rng(42).integers(0, 10000, size=args.rows, dtype=np.int64)
    table = pa.table({'amount': values})
    expected = int(values.sum())
    fixed_rg, fixed_page = 2_097_152, args.row_group_page_bytes
    layouts = {(rg, fixed_page) for rg in args.row_group_sizes} if args.sweep != 'pages' else set()
    if args.sweep in ('both', 'pages'):
        layouts |= {(fixed_rg, page) for page in (16_384, 65_536, 262_144, 1_048_576, 16_777_216)}
    layouts = sorted(layouts)
    manifest = []
    for rg, page in layouts:
        path = (args.data_dir / f'rg{rg}_page{page}.parquet').resolve()
        pq.write_table(table, path, row_group_size=rg, data_page_size=page,
                       write_batch_size=1024, use_dictionary=False, compression='NONE',
                       column_encoding='PLAIN', data_page_version='1.0', write_statistics=False)
        metadata = fastparquet.ParquetFile(path)
        counts = []
        for group in metadata.row_groups:
            stats = group.columns[0].meta_data.encoding_stats
            if stats is None:
                raise RuntimeError('Writer did not record page counts')
            counts.append(sum(s.count for s in stats if s.page_type in (0, 3)))
        manifest.append(dict(file=str(path), row_group_rows=rg, page_target_bytes=page,
                             row_groups=len(counts), data_pages=sum(counts),
                             pages_per_chunk_min=min(counts), pages_per_chunk_max=max(counts),
                             file_bytes=path.stat().st_size))
        print(f'Generated {path.name}: {len(counts)} row groups, {sum(counts)} pages', flush=True)
    environment = dict(os.environ, KVIKIO_COMPAT_MODE='ON')
    result = subprocess.run([str(args.binary.absolute()), str(args.repeats),
                             *[m['file'] for m in manifest]], env=environment,
                            text=True, capture_output=True, check=True)
    (args.output / 'samples.csv').write_text(result.stdout)
    (args.output / 'stderr.log').write_text(result.stderr)
    samples = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(samples) != len(manifest) * args.repeats:
        raise RuntimeError('Unexpected number of benchmark samples')
    for sample in samples:
        if int(sample['rows']) != args.rows or int(sample['sum']) != expected:
            raise RuntimeError(f'Incorrect GPU result: {sample}')
    for layout in manifest:
        measurements = [s for s in samples if s['file'] == layout['file']]
        times = [float(s['read_ms']) for s in measurements]
        layout.update(read_ms_median=statistics.median(times), read_ms_min=min(times),
                      read_ms_max=max(times), sum_ms_median=statistics.median(float(s['sum_ms']) for s in measurements))
        layout['effective_gib_s'] = args.rows * 8 / (1024**3) / (layout['read_ms_median'] / 1000)
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                       '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = 'unavailable'
    context = dict(rows=args.rows, expected_sum=expected, repeats=args.repeats,
                   seed=42, sweep=args.sweep, row_group_page_bytes=fixed_page, gpu=gpu, kernel=platform.release(), pyarrow=pa.__version__,
                   numpy=np.__version__, fastparquet=fastparquet.__version__,
                   kvikio_compat_mode='ON', layouts=manifest)
    (args.output / 'manifest.json').write_text(json.dumps(context, indent=2) + '\n')
    with (args.output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    lines = ['# Parquet layout experiment', '',
             f'{args.rows:,} identical non-null INT64 values; GPU sum verified as {expected:,} in every sample.',
             f'GPU / driver: {gpu}. Kernel: {platform.release()}.',
             f'{args.repeats} measured rounds; one unmeasured warm-up per file; shuffled file order each round.',
             'Uncompressed PLAIN encoding, no dictionary or statistics. KVIKIO_COMPAT_MODE=ON (no GDS).', '',
             'Times include metadata, file access, transfer, allocation and GPU decode, ending with GPU synchronization.',
             'Sum is timed separately. CUDA initialization and table destruction are outside read timing.',
             f'Storage location: {args.data_dir.resolve()}. Files are warmed before measurement; no cache eviction.', '',
             '| Rows/group | Page target (KiB) | Groups | Total pages | Pages/chunk | Median read (ms) | Min–max (ms) | Effective GiB/s |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in manifest:
        lines.append(f"| {m['row_group_rows']:,} | {m['page_target_bytes']//1024} | {m['row_groups']} | {m['data_pages']} | {m['pages_per_chunk_min']}–{m['pages_per_chunk_max']} | {m['read_ms_median']:.3f} | {m['read_ms_min']:.3f}–{m['read_ms_max']:.3f} | {m['effective_gib_s']:.2f} |")
    lines += ['', ('Page sweep: rows/group = 2,097,152.' if args.sweep == 'pages' else
               f'Row-group sweep: page target = {fixed_page // 1024:,} KiB. ' +
               ('Page sweep: compare rows/group = 2,097,152.' if args.sweep == 'both' else 'Only row-group size varies.')),
              'Page targets are approximate; counts above come from actual Parquet encoding statistics.',
              'Effective bandwidth counts decoded amount bytes; it is not physical storage throughput.',
              'This is a small single-column layout study, not a reproduction of the paper’s full TPC-H/GDS benchmark.',
              'Repeated rounds share one GPU process; medians and ranges are descriptive, not confidence intervals.']
    report = '\n'.join(lines) + '\n'
    (args.output / 'report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
