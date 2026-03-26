# Core-SG

[![PyPI version](https://img.shields.io/pypi/v/core-sg.svg)](https://pypi.org/project/core-sg/)
[![Python versions](https://img.shields.io/pypi/pyversions/core-sg.svg)](https://pypi.org/project/core-sg/)
[![Tests](https://img.shields.io/github/actions/workflow/status/midas-core-sg/core-sg/test.yml?branch=develop&label=tests)](https://github.com/midas-core-sg/core-sg/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Core-SG - Core Support Graph for efficient computation of multiple MSTs and HDBSCAN-style hierarchy outputs over varying values of `k`. Core-SG builds reusable graph support at `k_max` and then extracts minimum spanning trees and hierarchy artifacts for smaller values of `k <= k_max`. This allows Core-SG to support repeated multi-`k` analysis more efficiently than rebuilding the full structure for each `k`.

In practice this means that Core-SG lets you fit once at `k_max` and reuse the result straight away for many smaller `k` values with little or no extra setup, while keeping familiar HDBSCAN-like outputs such as `labels_`, `probabilities_`, `cluster_persistence_`, `condensed_tree_`, `single_linkage_tree_`, and `minimum_spanning_tree_`.

Core-SG is ideal for exploratory multi-`k` density-based analysis; it is a practical approach for workflows where you want to compare smoothing levels on the same dataset and inspect graph-level artifacts, not only final labels.

Based on the papers:


>Antonio Cavalcante Araujo Neto, Murilo Coelho Naldi, Ricardo J. G. B. Campello, and Jorg Sander. CORE-SG: Efficient Computation of Multiple MSTs for Density-Based Methods. In: 2022 IEEE 38th International Conference on Data Engineering (ICDE), IEEE, pp. 951-964. 2022.



>Leland McInnes and John Healy. Accelerated Hierarchical Density Based Clustering. In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp. 33-42. 2017.



>R. Campello, D. Moulavi, and J. Sander. Density-Based Clustering Based on Hierarchical Density Estimates. In: Advances in Knowledge Discovery and Data Mining, Springer, pp. 160-172. 2013.


Documentation and project overview are available in this repository. Notebooks comparing Core-SG to HDBSCAN and illustrating the intended multi-`k` workflow are available in [`notebooks/`](notebooks/).

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

Install with development tooling:

```bash
pip install -e ".[dev]"
```

Dependencies:

- `numpy>=1.24,<3`
- `pandas>=2.0`
- `scikit-learn>=1.3`
- `hdbscan>=0.8.39`

The package metadata, runtime dependencies, and optional extras are defined in `pyproject.toml`.

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

If you use Core-SG in scientific or technical work, please cite the Core-SG paper:

```bibtex
@inproceedings{neto2022core_sg,
  author = {Neto, Antonio Cavalcante Araujo and Naldi, Murilo Coelho and Campello, Ricardo J. G. B. and Sander, Jorg},
  title = {{CORE-SG}: Efficient Computation of Multiple MSTs for Density-Based Methods},
  booktitle = {2022 IEEE 38th International Conference on Data Engineering (ICDE)},
  pages = {951--964},
  year = {2022},
  publisher = {IEEE},
  doi = {10.1109/ICDE53745.2022.00076},
  url = {https://doi.org/10.1109/ICDE53745.2022.00076}
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
