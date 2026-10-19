# Compressed multi-row-group subpass benchmark

805,306,368 rows; 192 row groups; three selected integer columns; ZSTD level 3; 1 MiB page target.
File size 6038.8 MiB; fixed-width decoded values alone 15360 MiB, plus validity masks.
All 18 baseline and 6 profiled sums matched 40,264,940,585,830.
3 baseline rounds and 1 separate traced rounds; warm cache, shuffled configs, one excluded warm-up per configuration.

| Pass hint (MiB) | Output hint (MiB) | Input passes | Subpasses | Output chunks | Read median (ms) | Read min–max (ms) | Sum median (ms) | Total median (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 96 | 288 | 288 | 10364.20 | 10288.40–10397.50 | 74.15 | 10633.40 |
| 512 | 0 | 48 | 144 | 144 | 6646.10 | 6644.28–6671.09 | 45.50 | 6822.12 |
| 1024 | 0 | 22 | 64 | 64 | 4553.30 | 4514.46–4558.60 | 26.69 | 4706.61 |
| 2048 | 0 | 11 | 31 | 31 | 3690.11 | 3676.79–3698.85 | 14.91 | 3850.50 |
| 4096 | 0 | 5 | 15 | 15 | 3437.42 | 3431.35–3441.63 | 11.53 | 3580.63 |
| 8192 | 0 | 3 | 8 | 8 | 3347.94 | 3322.66–3356.36 | 9.27 | 3482.58 |

Zero means no limit. Both parameters are hints, not strict caps on GPU memory.
Pass sweep: output hint is 0 (unlimited). Output sweep: pass hint fixed at 256 MiB.
Read time sums synchronized constructor, has_next and read_chunk calls. has_next can prepare/decompress data and must be timed.
Sum time includes per-output-chunk GPU sum and scalar transfer. Total also includes buffer/reader destruction and loop overhead.
Only amount is summed, but all three integer columns are decoded and every output table must have three columns.
Input-pass counts use count_page_headers_kernel as a proxy, validated by this fixture having no page indexes. Subpass counts use preprocess_subpass_pages ranges.
Output counts come from returned tables and are cross-checked against read_chunk_internal ranges in each traced sample.
Fewer/smaller output tables are not evidence of more subpasses. Trace counts distinguish those boundaries.
A pass limit below one compressed row-group size can still load that group and exceed the requested hint.
Changing the pass hint can change input-pass and subpass counts together; do not attribute all differences solely to subpass count.
The dataset is synthetic but includes multiple row groups, three flat integer columns with no null values or nesting.
There are no filters or GDS, and no hardware utilization counters are collected.
