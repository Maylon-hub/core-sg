# CoreSG: Extracting Multiple Density Minimum Spanning Trees
core-sg is a Python library for building a Core-SG graph and reusing it to recover minimum spanning trees (MSTs) and HDBSCAN-style hierarchies for different values of k without rebuilding the full graph from scratch every time.

> Important acknowledgment
>
> This project is strongly inspired by and partially built on top of the public hdbscan package. The current implementation directly uses public and internal functionality from hdbscan, including:
>
> - hdbscan.HDBSCAN
> - hdbscan.hdbscan_._tree_to_labels
> - hdbscan._hdbscan_linkage.label
> - hdbscan.plots.CondensedTree
> - hdbscan.plots.SingleLinkageTree
> - hdbscan.plots.MinimumSpanningTree
>
> For that reason, hdbscan must be cited and credited as a core reference for this project.

---

## Overview

The main idea of core-sg is to build a graph structure once at a reference value k_max and then reuse that structure to:

1. recover the MST associated with a smaller k
2. recover the corresponding HDBSCAN-style hierarchy for that k
3. expose the same kinds of hierarchy objects users already know from hdbscan

This makes core-sg especially useful when you want to explore or compare multiple k values while preserving a workflow that remains close to HDBSCAN.

---

## Core concepts

At a high level, the current implementation on the develop branch works as follows:

1. compute the pairwise distance matrix D
2. extract the k_max nearest-neighbor graph
3. compute the list of core distances up to k_max
4. run an HDBSCAN reference fit at k_max to obtain the reference MST
5. merge the required MST edges into the graph support structure
6. build the Core-SG
7. for any chosen k <= k_max, reweight the graph using the mutual reachability rule
8. recover the MST with Kruskal
9. recover the hierarchy using HDBSCAN internals

The library currently exposes both:

- low-level functions for graph construction and MST recovery
- a higher-level CoreSG class that caches the reference fit and exposes HDBSCAN-style artifacts

---

## Visual overview

![Core-SG pipeline](docs/images/coresg_pipeline.png)

![CoreSG class lifecycle](docs/images/coresg_class_lifecycle.png)

---

## Installation

```bash
pip install core-sg
```

For local development:

```bash
pip install -e .
```

Current runtime dependencies include:

- numpy
- pandas
- scikit-learn
- hdbscan

---

## Basic usage

The main public entry point is the CoreSG class.

```python
import numpy as np
from sklearn.datasets import make_blobs

from core_sg import CoreSG

X, _ = make_blobs(
    n_samples=1000,
    n_features=10,
    centers=10,
    random_state=42,
)

core = CoreSG(
    metric="euclidean",
    p=2,
    debug=False,
    cluster_selection_method="eom",
    allow_single_cluster=False,
)

core.fit(X, k_max=15)

# Recover the MST for a chosen k
mst_k10 = core.extract_mst_from_core_sg(k=10)

# Recover the hierarchy for the same k
core.extract_hierarchy_from_core_sg(k=10)

labels = core.labels_
probabilities = core.probabilities_
cluster_persistence = core.cluster_persistence_

condensed_tree = core.condensed_tree_
single_linkage_tree = core.single_linkage_tree_
minimum_spanning_tree = core.minimum_spanning_tree_
```

If you want the tree objects that were cached during the reference fit at k_max, you can retrieve them directly:

```python
fitted = core.get_fitted_hdbscan_objects(wrapped=True)

labels_k_max = fitted["labels_"]
probabilities_k_max = fitted["probabilities_"]
cluster_persistence_k_max = fitted["cluster_persistence_"]
condensed_tree_k_max = fitted["condensed_tree_"]
single_linkage_tree_k_max = fitted["single_linkage_tree_"]
minimum_spanning_tree_k_max = fitted["minimum_spanning_tree_"]
```

---

## Getting more information about a fit

Following the spirit of the hdbscan documentation, core-sg exposes not only labels but also the hierarchy artifacts behind them.

### Condensed tree

After extracting a hierarchy, the condensed_tree_ property returns an hdbscan.plots.CondensedTree wrapper:

```python
core.extract_hierarchy_from_core_sg(k=10)
condensed = core.condensed_tree_
condensed_df = condensed.to_pandas()
```

### Single linkage tree

The single linkage structure can also be accessed in a wrapper compatible with the HDBSCAN plotting API:

```python
slt = core.single_linkage_tree_
slt_df = slt.to_pandas()
```

### Minimum spanning tree

The MST recovered from Core-SG is also exposed through the HDBSCAN wrapper when raw input data is available:

```python
mst = core.minimum_spanning_tree_
mst_df = mst.to_pandas()
```

For the reference fit at k_max, the equivalent accessors are:

- condensed_tree_k_max_
- single_linkage_tree_k_max_
- minimum_spanning_tree_k_max_

---

## How Core-SG works

This section mirrors the explanatory style commonly used in the hdbscan docs.

### 1. Build the distance matrix

The build_core_sg_from_data function starts from X and computes a pairwise distance matrix D using sklearn.metrics.pairwise_distances.

It currently supports:

- general pairwise metrics supported by scikit-learn
- a special handling for metric = minkowski with parameter p
- a compatibility path for metric = arccos via cosine distance

### 2. Build the k-nearest-neighbor support

The function knn_from_precomputed extracts the nearest neighbors of each point directly from the precomputed matrix D, and build_knng_vectors turns those neighbors into:

- metric_edges: edges stored as bigger, smaller, dist
- knng_to_insert: the graph-format edges stored as i, neighbor, dist

### 3. Compute core-distance information up to k_max

Instead of storing only one core-distance vector, the implementation stores a full core_k_list, where each column corresponds to a valid k.

This makes it possible to reuse the same Core-SG for multiple values of k <= k_max.

### 4. Get the reference MST at k_max

The current implementation calls a reference HDBSCAN fit with:

- metric = precomputed
- algorithm = generic
- approx_min_span_tree = False
- gen_min_span_tree = True
- match_reference_implementation = True

This reference fit is used to recover the MST associated with the mutual reachability graph at k_max.

### 5. Merge support edges and sort the graph

The MST edges needed for lookup are merged into metric_edges, and the graph representation is assembled into the final core_sg.

The edge weights inside core_sg are initially placeholders and are overwritten later during reweighting.

### 6. Reweight for a target k

For a chosen k, the function reweight_core_sg_mutual_reachability updates each edge weight using the standard mutual reachability rule:

```
w(u, v) = max(core_k[u], core_k[v], dist(u, v))
```

### 7. Recover the MST

Once the graph is reweighted for a target k, the function kruskal_mst builds the MST.

The current implementation uses a Union-Find structure with:

- path compression / path halving
- union by rank
- stable edge sorting by weight

### 8. Recover the hierarchy

For a target k, the class method extract_hierarchy_from_core_sg:

1. gets the MST
2. calls label from HDBSCAN internals to build the single linkage tree
3. calls _tree_to_labels through the local helper tree_to_labels
4. stores:
   - labels_
   - probabilities_
   - cluster_persistence_
   - condensed_tree_
   - single_linkage_tree_
   - minimum_spanning_tree_

For k == k_max, the class simply reuses the already cached HDBSCAN artifacts from the reference fit.

---

## Public API

## High-level class

### CoreSG(metric = euclidean, p = 2, debug = False, **hdbscan_kwargs)

Main class for building and reusing a Core-SG graph.

#### Main constructor arguments

- metric: distance metric used during pairwise distance computation
- p: power parameter for metrics such as Minkowski
- debug: prints timing information when True
- **hdbscan_kwargs: optional HDBSCAN-related parameters reused when compatible with _tree_to_labels

#### Main methods

##### fit(X, k_max, test_only = False)

Builds the Core-SG once and stores the HDBSCAN reference artifacts for k_max.

##### extract_mst_from_core_sg(k, toDF = False)

Returns:

- the cached MST if k == k_max
- a recomputed MST from Core-SG if k < k_max

If toDF = True, returns a pandas.DataFrame with columns:

- to
- from
- weight

##### extract_hierarchy_from_core_sg(k)

Reconstructs the HDBSCAN-style hierarchy for a chosen k.

If k == k_max, the method reuses the cached fit outputs.

##### get_core_distance(k)

Returns the core-distance vector associated with a chosen k.

##### get_core_sg_mutual_reachability_distance(k)

Returns the Core-SG graph after reweighting with mutual reachability distance for the chosen k.

##### get_fitted_hdbscan_objects(wrapped = True)

Returns the HDBSCAN artifacts stored during the reference fit at k_max.

When wrapped = True, the tree artifacts are returned as HDBSCAN wrapper objects.

---

## Tree and graph properties

After a hierarchy is extracted, the following properties become available:

- condensed_tree_
- single_linkage_tree_
- minimum_spanning_tree_

For the reference fit at k_max, the cached versions are:

- condensed_tree_k_max_
- single_linkage_tree_k_max_
- minimum_spanning_tree_k_max_

These properties intentionally follow the access pattern familiar to users of hdbscan.

---

## Low-level functions

The current branch also exposes a set of low-level functions used internally by the class.

### Graph construction

- build_core_sg_from_data
- knn_from_precomputed
- build_knng_vectors
- add_mst_edges_to_metric_edges

### Reweighting

- reweight_core_sg_mutual_reachability

### MST recovery

- kruskal_mst
- mst_from_core_sg

### Hierarchy recovery

- tree_to_labels

---

## Relationship with HDBSCAN

core-sg should be understood as a project that is compatible with and structurally inspired by HDBSCAN, not as a replacement for it.

The current implementation depends on hdbscan in at least four ways:

1. reference behavior
   - the reference fit at k_max is computed with hdbscan.HDBSCAN

2. hierarchy conversion
   - _tree_to_labels is used to obtain labels, probabilities, persistence, and condensed tree

3. single-linkage conversion
   - label is used to transform the MST into the single-linkage structure

4. plot-compatible wrappers
   - CondensedTree
   - SingleLinkageTree
   - MinimumSpanningTree

Because of this, any public-facing documentation, paper, presentation, or package page for core-sg should explicitly acknowledge hdbscan as a foundational reference.

---

## Validation

The current develop branch includes a validation script in:

tests/validate_core_sg.py

That script is focused on comparing Core-SG behavior against a reference HDBSCAN execution on synthetic data. In particular, it validates whether:

- the HDBSCAN MST is contained in the Core-SG support graph
- the expected mutual reachability weights match the HDBSCAN reference formulation
- the MST recovered from Core-SG matches the HDBSCAN MST in edges and weights

A typical execution pattern is:

```
python tests/validate_core_sg.py --n 5000 --d 10 --centers 10 --k 15 --seed 42
```

---

## Current limitations

At the time of writing, the develop branch is still an early public version of the project.

Notable points to keep in mind:

- the repository README.md is still a placeholder
- the current validation on develop is script-based rather than a full published documentation workflow
- the implementation currently depends on HDBSCAN internals, which means compatibility should be checked whenever the hdbscan version changes

---

## When to use Core-SG

core-sg is useful when:

- you want to evaluate several values of k from the same graph support
- you want to reuse one reference fit instead of rebuilding everything independently
- you want HDBSCAN-style labels and hierarchy objects while keeping control over graph construction and MST recovery

---

## Citing and crediting dependencies

If you use core-sg in a report, paper, notebook, benchmark, or presentation, please also cite and credit:

- the hdbscan library
- the HDBSCAN documentation
- the hdbscan source code components reused by this project

This is especially important because the current implementation directly relies on HDBSCAN internals for hierarchy conversion and plotting-compatible tree wrappers.

---

## References

### Core-SG repository

- Repository:
  - https://github.com/midas-core-sg/core-sg
- Develop branch:
  - https://github.com/midas-core-sg/core-sg/tree/develop

### HDBSCAN documentation

- Basic usage:
  - https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
- Getting more information about a clustering:
  - https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html
- How HDBSCAN works:
  - https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
- API reference:
  - https://hdbscan.readthedocs.io/en/latest/api.html

### HDBSCAN source code

- hdbscan.hdbscan_:
  - https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/hdbscan/hdbscan_.py
- hdbscan.plots:
  - https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/hdbscan/plots.py

### Reference note

This project uses HDBSCAN both as:

- a conceptual reference for hierarchy construction and inspection
- a direct implementation dependency for part of the current pipeline