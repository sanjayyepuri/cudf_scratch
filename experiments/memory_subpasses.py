"""Separate memory-instrumented runs; retain original timings and trace counts."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MiB = 1024**2


def save_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/results-subpasses')
    args = parser.parse_args()
    output = args.output
    fixture = json.loads((output/'fixture.json').read_text())
    baseline = list(csv.DictReader((output/'summary.csv').open()))
    limits = [f"{int(r['pass_limit_mib'])*MiB}:{int(r['chunk_limit_mib'])*MiB}" for r in baseline]
    command = [str(ROOT/'bazel-bin/subpass_benchmark'), str(args.repeats), fixture['file'], *limits]
    if not args.analyze_only:
        env = dict(os.environ, CUDF_MEMORY='1', KVIKIO_COMPAT_MODE='ON')
        env.pop('CUDF_PROFILE', None)
        env.pop('CUDF_BENCH_SINGLE_COLUMN', None)
        with (output/'memory-samples.csv').open('w') as raw, (output/'memory.log').open('w') as log:
            subprocess.run(command, env=env, stdout=raw, stderr=log, check=True)
        (output/'memory-capture.json').write_text(json.dumps(dict(
            command=command, repeats=args.repeats, warmups=1,
            allocator='RMM statistics adaptor over original default cuda_memory_resource; no pool',
            device_probe='cudaMemGetInfo, requested 1 ms polling interval; device-wide sampled usage, not physical-residency proof',
            timing='Keep original uninstrumented baseline; memory-run timings include probe overhead'), indent=2)+'\n')
    samples = list(csv.DictReader((output/'memory-samples.csv').open()))
    if len(samples) != args.repeats * len(baseline):
        raise RuntimeError('Missing memory samples')
    for r in samples:
        if (int(r['sum']) != fixture['expected_sum'] or int(r['rows']) != fixture['rows']
                or int(r['columns']) != fixture['columns'] or int(r['rmm_final_bytes']) != 0
                or int(r['device_sample_failures']) != 0 or int(r['device_samples']) < 1):
            raise RuntimeError(f'Memory/correctness validation failed: {r}')
    summary = []
    for b in baseline:
        group = [r for r in samples if int(r['pass_limit']) == int(b['pass_limit_mib'])*MiB
                 and int(r['chunk_limit']) == int(b['chunk_limit_mib'])*MiB]
        if len(group) != args.repeats or any(int(r['output_chunks']) != int(float(b['output_chunks_median'])) for r in group):
            raise RuntimeError('Configuration/output counts differ from baseline')
        item = {k: b[k] for k in ('pass_limit_mib','chunk_limit_mib','input_passes','subpasses','read_ms_median')}
        for key in ('rmm_peak_bytes','rmm_total_bytes','device_start_bytes','device_sampled_peak_bytes'):
            values = [int(r[key]) for r in group]
            item[key+'_median'] = statistics.median(values)
            item[key+'_min'] = min(values)
            item[key+'_max'] = max(values)
        item['device_sampled_increase_bytes_median'] = statistics.median(
            int(r['device_sampled_peak_bytes'])-int(r['device_start_bytes']) for r in group)
        summary.append(item)
    save_csv(output/'memory-summary.csv',summary)
    lines = ['# GPU memory versus reader batching', '',
             f"All {len(samples)} measured reads matched the expected sum and row count. All ended with zero tracked live bytes and no device-query failures.",
             '', '| Pass / output hint (MiB) | Input passes / subpasses | Baseline read (ms) | Peak live RMM (MiB) | Sampled device increase (MiB) |',
             '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['pass_limit_mib']} / {r['chunk_limit_mib']} | {r['input_passes']} / {r['subpasses']} | {float(r['read_ms_median']):.2f} | {r['rmm_peak_bytes_median']/MiB:.2f} | {r['device_sampled_increase_bytes_median']/MiB:.2f} |")
    lines += ['', 'Zero disables a hint. Timings and trace counts are from the original separate runs.',
              'Peak RMM bytes are live allocation requests routed through the current RMM resource, including reader temporaries, output columns and reduction scratch. The original default CUDA resource is wrapped, without adding a pool. Buffers are released between returned tables.',
              f'{args.repeats} measured memory rounds per configuration, shuffled with seed 42, after one excluded warm-up per configuration.',
              'Device usage is total minus free from cudaMemGetInfo, sampled on a separate thread with a requested 1 ms sleep. The observed peak is a lower bound: short-lived allocations may be missed. It includes the CUDA context, driver/library allocations and other GPU users; its increase is relative to each sample start.',
              'Neither allocator accounting nor cudaMemGetInfo proves physical residency under WSL or measures process-exclusive VRAM. Async frees and future pool allocators can separate logical live bytes from physically reserved memory. These hints remain soft targets, not capacity guarantees.',
              'Use peak live bytes to compare working sets, allow headroom for untracked allocations and concurrent workloads, and measure again if retaining output tables or changing allocator/workload.', '',
              '![Peak live GPU allocations versus baseline reader time](memory.png)', '',
              '[Raw memory samples](memory-samples.csv) · [Joined summary](memory-summary.csv) · [Capture setup](memory-capture.json)']
    (output/'memory-report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = [r for r in summary if r['chunk_limit_mib']=='0']
    secondary = [r for r in summary if r['pass_limit_mib']=='64']
    has_output_sweep = len({r['chunk_limit_mib'] for r in secondary}) > 1
    fig, axes = plt.subplots(1, 2 if has_output_sweep else 1,
                             figsize=(12 if has_output_sweep else 7.5, 4.8),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    panels = [(axes[0], primary, 'Pass hint sweep; output unlimited')]
    if has_output_sweep:
        panels.append((axes[1], secondary, 'Output hint sweep; pass hint 64 MiB'))
    for ax, rows, title in panels:
        for r in rows:
            x = r['rmm_peak_bytes_median']/MiB
            y = float(r['read_ms_median'])
            label = (r['pass_limit_mib'] if ax is axes[0] else r['chunk_limit_mib']) + ' MiB'
            if label == '0 MiB': label = 'Unlimited'
            ax.scatter(x, y, s=60)
            ax.annotate(label, (x, y), xytext=(5, 8), textcoords='offset points', fontsize=9)
        ax.set_title(title)
        ax.set_xlabel('Peak live RMM allocations (MiB)')
        ax.set_ylabel('Baseline reader time (ms)')
        ax.grid(alpha=.25)
        ax.margins(x=.3, y=.3)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    fig.suptitle('GPU working-set / latency tradeoff: compressed integer reader\nMemory measured separately; timings and pass counts retained from baseline')
    fig.savefig(output/'memory.png',dpi=180)
    fig.savefig(output/'memory.svg')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
