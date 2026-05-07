Benchmark Methodology
=====================

Core-SG performance should be interpreted as a build-once, extract-many
workflow. The initial build can be more expensive than a single HDBSCAN run,
but repeated extraction for multiple ``k`` values can amortize that cost.

The benchmark material in ``benchmarking/`` evaluates repeated multi-``k``
workloads over synthetic datasets. The central comparison is cumulative time
for all requested ``k`` values, not only the first result.

The report assets are copied into the documentation by
``docs/scripts/generate_benchmark_figures.py``.
