# Diagnostic small-hint timing sweep (15 GiB)

All 15 unprofiled measured reads passed the expected sum and row/column counts.
The pass hints were 16, 64, 256, 512 and 1,024 MiB; output hint disabled.
Three measured rounds after an excluded warm-up per configuration.

The trace capture was stopped when the experiment was redirected to a more
representative 256 MiB–8 GiB grid. No complete trace or memory measurements are
claimed for this diagnostic sweep. samples.csv and baseline.log preserve the
completed measurements; the primary sweep is ../results-large-15gib/.
