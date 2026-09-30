# Benchmarking artifacts

The scripts, configurations, and scientific reports here are versioned so the
experiments can be reproduced. Complete execution outputs are normally not
versioned; rerun the corresponding scripts to regenerate them. Temporary logs,
per-run telemetry, and intermediate results are ignored by Git.

A small set of aggregated CSVs and manifests remains in `results*` because the
runtime reports, quality documentation, or `paper/main.tex` cite or analyze
those files directly. Four report figures live in `run_time/report_assets/`;
the connectivity figures used by the documentation live in
`missing_edges/figures/`. Do not treat the retained results as a complete
archive of all benchmark runs. The `large_grid_manifest.json` command entries
use a repository-relative Python executable path to avoid recording a
contributor's private checkout location; experimental parameters and measured
values were not changed.
