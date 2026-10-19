# Large-scale refresh for the Parquet article

This follow-up targets **15 GiB of decoded integer values** per dataset
(16,106,127,360 bytes; about 16.1 decimal GB). File sizes differ with compression.
It reuses the RTX 3090, prebuilt libcudf SDK, and synchronized C++ read/sum driver.

## Compressed memory-hint sweep

- 805,306,368 rows, three flat columns: INT64 amount, INT64 customer ID, INT32 quantity.
- 192 row groups of 4,194,304 rows: the same row-group size and distribution as the original 640 MiB fixture, with 24 times as many groups.
- ZSTD level 3, 1 MiB target Pages, no dictionary, nulls, nesting, or predicates.
- Pass hints: 256, 512, 1,024, 2,048, 4,096, 8,192 MiB; output hint disabled.
- Three baseline rounds after an excluded warm-up, one separate traced round after a warm-up, and two separate memory rounds after a warm-up.
- The unlimited configuration is omitted: the earlier 640 MiB fixture peaked at about 2.9 times its decoded value size, which extrapolates above the available 24 GiB GPU capacity at this scale. This is a capacity estimate, not an observed OOM.

Each output table is reduced and released before the next is retained. Reader time
includes the constructor, every `has_next()` and `read_chunk()`, and synchronization.
Reduction is timed separately. This does not benchmark an overlapping production pipeline.

```bash
bazelisk build //:subpass_benchmark
/tmp/cudf-layout-venv/bin/python experiments/subpasses.py \
  --groups 192 --group-rows 4194304 \
  --pass-limits-mib 256 512 1024 2048 4096 8192 \
  --output-limits-mib 0 --output-pass-mib 256 \
  --repeats 3 --trace-repeats 1 \
  --data-dir /tmp/cudf-large-data --output experiments/results-large-15gib
/tmp/cudf-layout-venv/bin/python experiments/memory_subpasses.py \
  --repeats 2 --output experiments/results-large-15gib
```

The generation streams one 80 MiB row group at a time; it never constructs the
entire 15 GiB table in host memory. Add `--reuse-fixture` to reuse a completed file.
The compressed file is about 5.9 GiB, so an excluded read can populate the host
filesystem cache. No cache flushing or cache-residency measurement is performed.

The completed initial 16–1,024 MiB timing sweep is preserved separately in
`results-large-small-hints-15gib/`. The 16 MiB hint is below one compressed Row
Group (about 31.5 MiB) and mainly demonstrates pathological batching overhead.
Its interrupted trace is not used as evidence; no memory peaks are claimed for
that diagnostic sweep. The primary grid emphasizes realistic working-set sizes.

## Uncompressed Page-size sweep

- 2,013,265,920 non-null INT64 values in 60 row groups of 33,554,432 rows.
- Every layout has the same values, RNG seed, and group boundaries.
- Uncompressed PLAIN encoding, no dictionary or statistics.
- Page targets: 16 KiB, 64 KiB, 256 KiB, 1 MiB, 16 MiB.
- Fixed 2 GiB pass hint, output hint disabled; only the Page target varies.
- Three baseline rounds after an excluded warm-up, plus one separate traced round after a warm-up.

The original 64 MiB Page study used whole-file reads and four row groups. At
15 GiB, encoded input plus output and scratch can exceed the device capacity,
so this new study deliberately uses bounded chunked reads and more row groups.
Compare Page sizes **within** a dataset; a cross-scale latency ratio would conflate
size, row-group layout, read API, and cache state. Page counts come from actual
Parquet encoding statistics, not the nominal target.

```bash
/tmp/cudf-layout-venv/bin/python experiments/large_pages.py
```

Fixture generation holds only one 256 MiB group at a time. The five files require
roughly 75 GiB of disk. Each 15 GiB file exceeds available host RAM; the excluded
warm-up does **not** establish a warm filesystem cache. File access and transfer
remain part of measured latency. Use `--reuse-fixture` to avoid regenerating files.

The optional `CUDF_BENCH_SINGLE_COLUMN=1` setting selects only amount in the same
C++ driver. Without it, the original three-column behavior and CSV schema remain.

## Trace and memory interpretation

- `preprocess_subpass_pages` counts subpasses; header-count kernel launches proxy input passes for these files without Page indexes.
- Returned table counts are checked against trace ranges.
- Page trace summaries include reader-phase kernels and copies across the entire read, excluding the per-table reduction phase. Timeline CSVs show an exact first-output-batch excerpt (its preparation and decoding), labelled separately from full-read totals.
- One profiled round provides descriptive trace timings, not stable profiled medians or confidence intervals.
- RMM reports peak logical live allocation requests through the current resource; no pool is introduced. Device-wide CUDA memory samples are reported separately.
- Neither metric proves process-exclusive physical residency under WSL. Hints remain soft targets, and output-retention policy changes capacity needs.
- Compatibility I/O is enabled. No GDS or hardware performance counters are used.

The scripts fail on any missing sample, row count, column count, or sum mismatch.
Baseline and memory runs remain separate so instrumentation overhead does not
enter the reported baseline timings. Original small-scale artifacts are preserved.

## Reports and article data

After both sweeps complete:

```bash
/tmp/cudf-layout-venv/bin/python experiments/large_report.py
/tmp/cudf-layout-venv/bin/python experiments/export_blog.py /path/to/blog-checkout
```

The exporter joins actual layouts by Page target and memory configurations by
both byte hints. It retains numeric precision and the original small-scale data;
only the UI rounds display values. The large timeline is a labelled first-output-
batch excerpt rather than a misleading forest of widened, subpixel events from
the entire read. Its kernel-cost totals still cover the full profiled read.

Large newly generated Nsight and SQLite captures are retained locally under
`experiments/traces/` and ignored by default. CSV summaries, validated raw samples,
first-batch timeline excerpts, setup metadata and figures are versioned.
