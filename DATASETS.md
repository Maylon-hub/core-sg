# Dataset provenance and distribution policy

The wheel contains no datasets. The sdist includes synthetic test/benchmark
generator code, but excludes raw CSV/NPY/NPZ files, notebooks with stored outputs,
and historical benchmark output directories. No source data is deleted locally.

- README examples: sklearn `make_blobs`, 1000x10, 10 centers, seed 42.
- Reference integration tests: `make_blobs`, 60x3, 4 centers, seed 42; connected
  SCORE-SG fixtures use seeds 42 and 123 (parameters remain in each test).
- Validation suite: `make_blobs`, 5000x10, 10 centers, seed 42.
- Public Euclidean regression: NumPy `default_rng`, normal arrays; seeds 37
  and 21, shapes 30x3 and 20x3.
- Unit fixtures: hand-constructed arrays in `tests/helpers.py`, no seed.
- RC smoke: `make_blobs`, 90x3, 3 centers, seed 42.

These are project-authored synthetic examples under BSD-3-Clause, using
[sklearn generators](https://scikit-learn.org/stable/datasets/sample_generators.html)
and NumPy. Record their versions, shape, distribution parameters and seed.
For other shipped test generators, the exact source parameters are authoritative;
randomized generators must carry an explicit seed.

Historical notebooks and benchmarking scripts can load real datasets through
sklearn/OpenML or user paths. Their outputs are **not RC-distributed data** and
their external licenses are not blanket-covered by this repository's license.
Redistribution needs dataset-by-dataset source/license approval before those
files become release examples. They remain preserved in Git; this exclusion
does not sanitize public Git history or authorize transfer of unknown data.

## Real loaders available in benchmark source (no raw data shipped)

| Loader | Original source / redistribution terms | Transformation |
|---|---|---|
| `load_iris` | [Fisher/UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), DOI 10.24432/C56C76, CC BY 4.0 | sklearn corrected features; targets kept separately for quality tests |
| `load_wine` | [Aeberhard/Forina UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), DOI 10.24432/C5PC7J, CC BY 4.0 | sklearn numeric features; benchmark preprocessing is recorded by each script |
| `load_breast_cancer` | [Wolberg/Mangasarian/Street UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), DOI 10.24432/C5DW2B, CC BY 4.0 | 30 diagnostic features, no subject ID |
| `load_digits` | [UCI optical recognition](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), Alpaydin/Kaynak, 1998, DOI 10.24432/C50P49, CC BY 4.0 | sklearn's 1,797-sample test subset, 8x8 images flattened; no copied dataset in this release |
| User CSV | User-supplied source/license; unknown by default | No redistribution authorization is inferred from loading a file |

All four UCI license pages were checked on 2026-09-27; retain attribution with
derived exports. Loaders have no seed;
synthetic benchmark seeds and generator parameters are explicit CLI inputs and
must be retained with outputs. Do not redistribute unknown user CSVs.
