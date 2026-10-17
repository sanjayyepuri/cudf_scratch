# Compressed multi-row-group subpass benchmark

33,554,432 rows; 8 row groups; three selected integer columns; ZSTD level 3; 1 MiB page target.
File size 251.6 MiB; fixed-width decoded values alone 640 MiB, plus validity masks.
All 40 baseline and 16 profiled sums matched 1,677,479,147,194.
5 baseline rounds and 2 separate traced rounds; warm cache, shuffled configs, one excluded warm-up per configuration.

| Pass hint (MiB) | Output hint (MiB) | Input passes | Subpasses | Output chunks | Read median (ms) | Read min–max (ms) | Sum median (ms) | Total median (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 | 153.10 | 150.52–162.71 | 0.82 | 160.70 |
| 16 | 0 | 8 | 248 | 248 | 6423.41 | 6384.54–6451.51 | 68.42 | 6557.15 |
| 64 | 0 | 8 | 32 | 32 | 998.97 | 993.45–1012.95 | 9.40 | 1031.34 |
| 64 | 8 | 8 | 32 | 128 | 1237.66 | 1236.86–1258.63 | 36.97 | 1339.24 |
| 64 | 32 | 8 | 32 | 32 | 1024.86 | 1022.31–1035.55 | 10.97 | 1056.93 |
| 64 | 128 | 8 | 32 | 32 | 1035.57 | 1022.14–1048.17 | 9.70 | 1068.61 |
| 256 | 0 | 4 | 12 | 12 | 462.27 | 454.49–467.92 | 3.90 | 474.98 |
| 512 | 0 | 2 | 6 | 6 | 295.13 | 291.77–304.58 | 2.36 | 304.23 |

Zero means no limit. Both parameters are hints, not strict caps on GPU memory.
Pass sweep: output hint is 0 (unlimited). Output sweep: pass hint fixed at 64 MiB.
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
