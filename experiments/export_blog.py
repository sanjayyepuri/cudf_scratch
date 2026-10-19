"""Join verified benchmark CSVs into the article's interactive data, without rounding."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MiB=1024**2


def read(path):return list(csv.DictReader(path.open()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('blog',type=Path)
    args=parser.parse_args()
    target=args.blog/'app/components/gpu-parquet/measurements.json'
    data=json.loads(target.read_text())
    old_trace=read(ROOT/'experiments/results-pages-traced/summary.csv')
    for p in data['pages']:
        t=next(t for t in old_trace if int(t['page_target_bytes'])==p['bytes'])
        p.update(grids=t['decoder_grid_blocks'],kernelCount=int(float(t['kernels_per_read'])),decoderLaunches=int(float(t['decoder_launches_per_read'])),traceEnd=32)
    pages_dir=ROOT/'experiments/results-large-pages-15gib'
    traces=read(pages_dir/'trace-summary.csv')
    data['largePages']=[]
    for row in read(pages_dir/'summary.csv'):
        page=int(row['page_target_bytes']);t=next(t for t in traces if int(t['page_target_bytes'])==page)
        events=read(pages_dir/f'timeline-{page}.csv')
        data['largePages'].append(dict(label=f'{page//1024} KiB' if page<MiB else f'{page//MiB} MiB',bytes=page,
            pages=int(row['data_pages']),perChunk=int(row['pages_per_chunk']),read=float(row['read_ms_median']),
            low=float(row['read_ms_min']),high=float(row['read_ms_max']),header=float(t['header_ms']),decoder=float(t['decoder_ms']),
            kernel=float(t['kernel_busy_ms']),memcpy=float(t['memcpy_busy_ms']),grids=t['decoder_grids'],
            kernelCount=int(t['kernel_count']),decoderLaunches=int(t['decoder_launches']),traceEnd=float(t['timeline_end_ms']),
            events=[[e['kind'],float(e['start_ms']),float(e['duration_ms'])] for e in events]))
    memory=read(ROOT/'experiments/results-large-15gib/memory-summary.csv')
    counts=read(ROOT/'experiments/results-large-15gib/summary.csv')
    data['largeRuns']=[]
    for m in memory:
        b=next(b for b in counts if b['pass_limit_mib']==m['pass_limit_mib'] and b['chunk_limit_mib']==m['chunk_limit_mib'])
        data['largeRuns'].append(dict(passHint=int(m['pass_limit_mib']),outputHint=int(m['chunk_limit_mib']),
            passes=int(m['input_passes']),subpasses=int(m['subpasses']),outputs=int(b['traced_output_chunks']),
            read=float(m['read_ms_median']),memory=float(m['rmm_peak_bytes_median'])/MiB))
    target.write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(f"Exported {len(data['largePages'])} page layouts and {len(data['largeRuns'])} memory configurations")


if __name__=='__main__':main()
