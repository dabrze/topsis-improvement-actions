# Perspektywy experiment package

This package reproduces the Perspektywy technical-university case study from
the five validated 2022--2026 editions.  It uses the seven `_rel` criterion
scores (each group leader is 100) and group weights for the relevant edition.

Run all experiments from the repository root:

```powershell
& 'C:\ProgramData\anaconda3\Scripts\conda.exe' run -n pad python experiments\perspektywy\run_all.py
& 'C:\ProgramData\anaconda3\Scripts\conda.exe' run -n pad python experiments\perspektywy\run_all.py --quick
```

`--quick` uses alpha values 0 and 0.5 for E2, 20 regular E4 repetitions, five
evolutionary E4 repetitions, evolutionary population 200, and five
generations. The full benchmark uses 200 regular repetitions, 30 evolutionary
repetitions, population 1000, and 200 generations. E3 survival, E2 margin
sweeps, and E2 values/weights ablation are enabled by default in `run_all`;
use `--skip-e3-survival`, `--skip-margins`, or `--skip-ablation` only for
development. Every script accepts `--focal`, `--year` (E1/E3/E4), `--pair YEAR
NEXT_YEAR` (E2), and `--outdir`.

Artifacts are written below `results/`, `tables/`, and `figures/`. E1 maps to
Sections 5.2--5.6, E2 to 5.7, E3 to 5.8, and E4 to 5.9. CSV files have
sidecar metadata with commit, configuration, UTC timestamp, and script name;
E4 additionally records runtime platform details.

Ranks use strict score comparison (ties receive the same strict rank), not
Perspektywy's published 0.5 p.p. ex-aequo rule. E2 applies a year-*t* delta
unchanged to the year-*t+1* relative-score scale and clips every criterion to
`[0, 100]`; this deliberately models the score-space information available to
a decision maker rather than raw institutional quantities.

E2's base problem uses the minimum strict-rank epsilon (`alpha = 0`). The
margin sweep additionally uses `max(1e-6, alpha * gap)`, where `gap` is the
aggregation-score distance to the alternative directly above the requested
target rank. This is a robustness extension, not a replacement for the
minimal-modification result. The published 0.5 p.p. ex-aequo band remains a
domain-grounded absolute sensitivity alternative.

The E2 ablation evaluates alpha-zero plans with complete next-edition values
and weights, next-edition values with source-edition weights, and
source-edition values with next-edition weights. For same-weight pairs, the
last control must survive at 100%; values-only results still conflate data
movement with indicator-composition changes.
