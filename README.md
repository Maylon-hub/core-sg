# CoreSG: Efﬁcient Computation of Multiple MSTs for Density-Based Methods

Core-SG graph construction and MST extraction utilities for HDBSCAN-style clustering.

core-sg is a Python library for building a reusable Core-SG graph and recovering minimum spanning trees and hierarchy objects for different values of k.

The goal is to stay close to the HDBSCAN workflow while giving more direct control over the graph and MST side of the pipeline.

## Important acknowledgment

This project is structurally inspired by and technically based on the `hdbscan` library.

In the current implementation, Core-SG relies on hdbscan for:

- reference behavior at k_max
- hierarchy post-processing
- single linkage tree conversion
- tree and MST wrapper objects

Because of that, any documentation, benchmark, report, or presentation involving Core-SG should explicitly cite and credit `hdbscan`.

## Installation

```
pip install core-sg
```

For local development:

```
pip install -e .
```

## Dependencies

Core-SG currently depends on:

- numpy
- pandas
- scikit-learn
- hdbscan

## Quick start

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

After fitting, the object stores the reference artifacts for k_max and can be reused for smaller values of k.

## Extracting an MST

```python
mst = core.extract_mst_from_core_sg(k=10)
```

If you want the MST as a DataFrame:

```python
mst_df = core.extract_mst_from_core_sg(k=10, toDF=True)
```

## Extracting a hierarchy

```python
core.extract_hierarchy_from_core_sg(k=10)

labels = core.labels_
probabilities = core.probabilities_
cluster_persistence = core.cluster_persistence_
```

## Inspecting clustering objects

Like HDBSCAN, Core-SG also exposes tree-style objects after hierarchy extraction.

### Condensed tree

```python
condensed_tree = core.condensed_tree_
```

### Single linkage tree

```python
single_linkage_tree = core.single_linkage_tree_
```

### Minimum spanning tree

```python
minimum_spanning_tree = core.minimum_spanning_tree_
```

## Accessing the reference fit at k_max

The artifacts saved during the reference fit can also be recovered directly.

```python
fitted = core.get_fitted_hdbscan_objects(wrapped=True)
```

This includes:

- labels_
- probabilities_
- cluster_persistence_
- condensed_tree_
- single_linkage_tree_
- minimum_spanning_tree_

There are also direct cached accessors for the k_max fit:

- condensed_tree_k_max_
- single_linkage_tree_k_max_
- minimum_spanning_tree_k_max_


## Why use Core-SG?

The library is built around a simple idea:

1. build the graph support once at k_max
2. reuse that support for smaller values of k
3. recover MSTs and HDBSCAN-style hierarchy objects without rebuilding everything from scratch

This makes it easier to compare multiple values of k in a consistent workflow.

Core-SG is useful when you want to:

- reuse a graph structure across several values of k
- recover MSTs without rebuilding the full support each time
- keep an HDBSCAN-like interface for labels and hierarchy objects
- work more directly with graph-level clustering structures


## Relationship with HDBSCAN

Core-SG is not intended to replace HDBSCAN.

Instead, it should be understood as a companion project that:

- follows an HDBSCAN-like user experience
- reuses HDBSCAN internals where appropriate
- focuses specifically on Core-SG graph construction and reuse across k values

If you are already familiar with HDBSCAN, the Core-SG interface should feel natural.

## License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.

## References

### Core-SG

- Repository:
  - https://github.com/midas-core-sg/core-sg
- Develop branch:
  - https://github.com/midas-core-sg/core-sg/tree/develop

### HDBSCAN

- Repository:
  - https://github.com/scikit-learn-contrib/hdbscan
- Documentation:
  - https://hdbscan.readthedocs.io/en/latest/
- Basic usage:
  - https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
- Getting more information:
  - https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html
- API reference:
  - https://hdbscan.readthedocs.io/en/latest/api.html


## Citation

If you use Core-SG in academic or technical work, please cite both the Core-SG paper and the relevant HDBSCAN references.

This project is structurally inspired by and technically dependent on hdbscan. In its current implementation, Core-SG reuses HDBSCAN concepts and parts of the HDBSCAN code path for hierarchy construction, tree conversion, and wrapper objects. For that reason, hdbscan should be acknowledged as a foundational reference whenever Core-SG is cited.

### Core-SG

- Neto, Antonio Cavalcante Araujo, Murilo Coelho Naldi, Ricardo J. G. B. Campello, and Jörg Sander. "CORE-SG: Efficient Computation of Multiple MSTs for Density-Based Methods." In Proceedings of the 2022 IEEE 38th International Conference on Data Engineering (ICDE), pp. 951-964. IEEE, 2022. DOI: 10.1109/ICDE53745.2022.00076.

### HDBSCAN

- McInnes, Leland, John Healy, and Steve Astels. "hdbscan: Hierarchical density based clustering." Journal of Open Source Software 2, no. 11 (2017): 205.

- McInnes, Leland, and John Healy. "Accelerated Hierarchical Density Based Clustering." In 2017 IEEE International Conference on Data Mining Workshops (ICDMW), pp. 33-42. IEEE, 2017.

- Bot, Daniël M., Jannes Peeters, Jori Liesenborgs, and Jan Aerts. "FLASC: a flare-sensitive clustering algorithm." PeerJ Computer Science 11 (2025): e2792. DOI: 10.7717/peerj-cs.2792.

