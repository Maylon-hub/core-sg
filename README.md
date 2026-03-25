# Core-SG

[![PyPI version](https://img.shields.io/pypi/v/core-sg.svg)](https://pypi.org/project/core-sg/)
[![Python versions](https://img.shields.io/pypi/pyversions/core-sg.svg)](https://pypi.org/project/core-sg/)
[![Tests](https://img.shields.io/github/actions/workflow/status/midas-core-sg/core-sg/test.yml?branch=main&label=tests)](https://github.com/midas-core-sg/core-sg/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Core-SG is a Python library for reusable graph support construction at `k_max`, followed by fast extraction of MSTs and HDBSCAN-style hierarchy outputs for smaller values of `k`.

The main goal is simple: fit once at `k_max`, reuse many times for `k <= k_max`.

## What Core-SG is for

Core-SG is designed for workflows where you need to compare multiple `k` values on the same dataset and you care about graph-level artifacts, not only final labels.

In practice, Core-SG helps you:

- reuse support across repeated `k` evaluations
- extract minimum spanning trees for different `k`
- keep an HDBSCAN-like workflow (`labels_`, `probabilities_`, `cluster_persistence_`)
- inspect tree artifacts (`condensed_tree_`, `single_linkage_tree_`, `minimum_spanning_tree_`)

## How to use Core-SG

Core-SG follows a two-stage workflow:

1. fit once with `k_max`
2. extract MSTs and hierarchies for smaller `k`

```python
from sklearn.datasets import make_blobs
from core_sg import CoreSG

X, _ = make_blobs(
    n_samples=1000,
    n_features=10,
    centers=10,
    random_state=42,
)

core = CoreSG(metric="euclidean", p=2)
core.fit(X, k_max=15)
```

### Extracting an MST

```python
mst = core.extract_mst_from_core_sg(k=10)
mst_df = core.extract_mst_from_core_sg(k=10, toDF=True)
```

### Extracting hierarchy outputs

```python
core.extract_hierarchy_from_core_sg(k=10)

labels = core.labels_
probabilities = core.probabilities_
cluster_persistence = core.cluster_persistence_
```

### Inspecting tree objects

```python
condensed_tree = core.condensed_tree_
single_linkage_tree = core.single_linkage_tree_
minimum_spanning_tree = core.minimum_spanning_tree_
```

### Accessing fitted artifacts at `k_max`

```python
fitted = core.get_fitted_hdbscan_objects(wrapped=True)
```

Returned keys:

- `labels_`
- `probabilities_`
- `cluster_persistence_`
- `condensed_tree_`
- `single_linkage_tree_`
- `minimum_spanning_tree_`

Direct cached wrappers at fit-time:

- `condensed_tree_k_max_`
- `single_linkage_tree_k_max_`
- `minimum_spanning_tree_k_max_`

## Performance (multi-k workflows)

Core-SG is optimized for repeated `k` analysis, not necessarily for a single one-off run.

In `notebooks/01-HDBSCAN_comparision.ipynb`, for a synthetic setup (`n=5000`, `d=2`, `centers=10`) with repeated evaluations from `k=30` down to `k=10`, cumulative runtime was:

- Core-SG: `9.76 s`
- HDBSCAN: `32.44 s`

This notebook demonstrates the intended tradeoff: higher upfront cost at `k_max`, lower cumulative cost when reusing across multiple smaller `k` values.

## Known limitations

- Core-SG provides strongest gains in repeated multi-`k` usage
- for single `k` workflows, plain HDBSCAN may be simpler
- current hierarchy pipeline still depends on HDBSCAN ecosystem components

## Installing

Install from PyPI:

```bash
pip install core-sg
```

Install for local development:

```bash
pip install -e .
```

Dependencies:

- `numpy>=1.24,<3`
- `pandas>=2.0`
- `scikit-learn>=1.3`
- `hdbscan>=0.8.39`

## Running tests

```bash
pytest tests -v -ra
```

## Python version

Core-SG supports Python `>=3.10`.

## Help and support

- Documentation and project overview: https://github.com/midas-core-sg/core-sg#readme
- Issues: https://github.com/midas-core-sg/core-sg/issues

## Contributing

Contributions are welcome. Please follow the contribution workflow in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgment

Core-SG is structurally inspired by and technically based on the `hdbscan` ecosystem.

- HDBSCAN repository: https://github.com/scikit-learn-contrib/hdbscan
- HDBSCAN documentation: https://hdbscan.readthedocs.io/en/latest/

## Citing

If you use Core-SG in scientific or technical work, please cite Core-SG and relevant HDBSCAN references.

```bibtex
@software{core_sg,
  title = {Core-SG},
  author = {Midas Core-SG Team},
  url = {https://github.com/midas-core-sg/core-sg}
}
```

## License

Core-SG is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## References


### HDBSCAN
- Repository: https://github.com/scikit-learn-contrib/hdbscan
- Documentation: https://hdbscan.readthedocs.io/en/latest/
- Basic usage: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
- Advanced usage: https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html
- API reference: https://hdbscan.readthedocs.io/en/latest/api.html