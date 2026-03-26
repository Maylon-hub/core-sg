# Testing Guide and Test Coverage Notes

This document clarifies how the project is currently tested, what each test category is intended to verify, and how contributors should run the checks locally.

Its goal is to replace fragmented or inconsistent test notes with a single reference written in consistent English and aligned with the current package entry points and dependencies.

## 1. Current testing objective

The current validation strategy for Core-SG is centered on the public behavior of the library:

- building the Core-SG support from input data
- extracting MSTs for a requested `k`
- reconstructing HDBSCAN-style hierarchy artifacts
- exposing cached objects from the reference fit at `k_max`
- handling invalid parameters consistently

Because the project is a graph- and MST-focused companion library for HDBSCAN-style workflows, the most important tests are the ones that confirm:

1. the reusable Core-SG structures are built correctly
2. extracted MSTs are structurally valid
3. hierarchy artifacts are exposed consistently
4. failure cases raise the expected errors instead of failing silently

## 2. Dependencies used to run tests locally

The repository metadata currently exposes a dedicated `test` extra in `pyproject.toml`, with:

- `pytest>=8.0`
- `pytest-cov>=5.0`

The development extra also includes:

- `ruff`

The runtime dependencies used by the library are declared in `pyproject.toml` and include:

- `numpy`
- `pandas`
- `scikit-learn`
- `hdbscan`

Python `>=3.10` is required by the package metadata.

## 3. Recommended local setup

From the repository root, install the package in editable mode together with the test dependencies:

```bash
pip install -e ".[test]"
```

If the development extra is preferred, this also works:

```bash
pip install -e ".[dev]"
```

Both approaches are consistent with the package metadata currently defined in `pyproject.toml`.

## 4. How to run tests locally

Run the full test suite from the repository root with:

```bash
pytest
```

To see coverage information:

```bash
pytest --cov=core_sg --cov-report=term-missing
```

To run the lint and formatting checks enforced by CI:

```bash
ruff check core_sg tests
ruff format --check core_sg tests
```

To validate the package build and metadata locally:

```bash
python -m build
python -m twine check dist/*
```

To run tests by category:

```bash
pytest tests/unit -q
pytest tests/integration -q
pytest tests/validation -q
```

To run tests by marker:

```bash
pytest -m unit -q
pytest -m integration -q
pytest -m "validation and not slow" -q
```

To run a single test file:

```bash
pytest path/to/test_file.py
```

To run a single test function:

```bash
pytest path/to/test_file.py -k test_name
```

## 5. Test Suite Structure

The suite is organized by responsibility so contributors can quickly identify the scope of each test group.

- `tests/unit/`
  Fast tests for isolated behavior. These use controlled fixtures and fake `hdbscan` modules when the goal is to validate internal behavior without relying on the real external dependency.
- `tests/integration/`
  Tests that exercise the package boundary and real integration with `hdbscan`. These confirm that the public API works end to end with the real dependency installed.
- `tests/validation/`
  Heavier behavioral validation tests that compare Core-SG outputs against HDBSCAN reference behavior across multiple values of `k`. These are closer to algorithm-equivalence checks than ordinary unit tests.
- `tests/helpers.py`
  Shared builders, payloads, and comparison utilities reused across the suite.
- `tests/validate.py`
  Validation helpers specifically used by the heavier equivalence-style tests.

This separation is meant to reduce navigation cost, make test intent clearer, and avoid mixing fast isolated tests with more expensive validation scenarios in the same directory.

## 6. What should be tested

The sections below describe the expected test coverage for the current codebase and what each test group is meant to validate.

### 6.1 Public import and package surface

These tests verify that the package exposes the public API expected by users.

What is tested:

- `from core_sg import CoreSG` works
- the package exports `CoreSG` through `core_sg/__init__.py`
- the library can be imported after editable installation without path hacks

Why it matters:

This is the first user-facing contract of the package. If the public import fails, the package is effectively unusable.

### 6.2 Core-SG build from raw data

These tests validate the behavior of `build_core_sg_from_data(...)` and the `fit(...)` path of `CoreSG`.

What is tested:

- valid input data produces Core-SG structures without errors
- the returned objects have consistent dimensions
- the distance matrix is built correctly from the chosen metric
- diagonal distances are zeroed
- `k_max` is stored correctly during `fit`
- the reference fit at `k_max` caches the expected artifacts

Why it matters:

This is the foundation of the whole library. Every later extraction depends on the correctness of this initial build step.

### 6.3 Validation of invalid parameters

These tests confirm that invalid inputs raise explicit errors.

What is tested:

- `n <= 1` is rejected
- `k_max <= 0` is rejected
- `k_max >= n` is rejected
- `k_max < 2` is rejected for HDBSCAN-compatible flows
- invalid `k` values passed to MST or hierarchy extraction are rejected

Why it matters:

The project should fail fast and clearly when the requested configuration is impossible or inconsistent.

### 6.4 MST extraction from Core-SG

These tests validate `mst_from_core_sg(...)` and `CoreSG.extract_mst_from_core_sg(...)`.

What is tested:

- MST extraction returns exactly `n - 1` edges
- returned edges are sorted by weight
- the MST is produced for `k <= k_max`
- extraction at `k == k_max` reuses the cached fit artifact
- optional DataFrame conversion via `toDF=True` returns typed columns

Why it matters:

MST extraction is one of the central promises of the library. It must be correct, reproducible, and convenient to inspect.

### 6.5 Mutual reachability reweighting behavior

These tests validate the reweighted graph returned by `get_core_sg_mutual_reachability_distance(...)` and related helpers.

What is tested:

- the reweighted graph is produced for a valid `k`
- the selected core distance really corresponds to column `k - 1` of `core_k_list`
- the reweighting step uses the shared Core-SG support rather than rebuilding the graph
- invalid `k` values raise errors

Why it matters:

This is the mechanism that enables reuse across multiple values of `k`, which is the main technical value proposition of Core-SG.

### 6.6 Hierarchy extraction and HDBSCAN-style outputs

These tests validate `extract_hierarchy_from_core_sg(...)` and the tree-to-label conversion flow.

What is tested:

- hierarchy extraction populates `labels_`
- hierarchy extraction populates `probabilities_`
- hierarchy extraction populates `cluster_persistence_`
- condensed tree, single linkage tree, and minimum spanning tree are populated
- extracting at `k == k_max` reuses the cached fit-time artifacts
- extracting at `k < k_max` builds the hierarchy from the extracted MST

Why it matters:

Users rely on Core-SG not only for MSTs but also for HDBSCAN-style downstream clustering artifacts.

### 6.7 Wrapped object accessors

These tests validate the property accessors that expose HDBSCAN-compatible wrapper objects.

What is tested:

- `condensed_tree_` raises an error before hierarchy extraction
- `single_linkage_tree_` raises an error before hierarchy extraction
- `minimum_spanning_tree_` raises an error before hierarchy extraction
- fit-time accessors such as `condensed_tree_k_max_` raise an error before `fit`
- wrapped objects are returned after the corresponding raw arrays are available

Why it matters:

The properties are part of the public interface and should fail predictably when called too early.

### 6.8 Cached fitted-object retrieval

These tests validate `get_fitted_hdbscan_objects(...)`.

What is tested:

- calling it before `fit` raises an error
- `wrapped=True` returns the wrapper-based representation
- `wrapped=False` returns the raw stored artifacts
- the returned dictionary contains all expected keys

Why it matters:

This method is a core part of the reusable workflow and should be reliable for both interactive usage and downstream tooling.

### 6.9 DataFrame conversion helper

These tests validate `_mst_to_dataframe(...)`.

What is tested:

- the DataFrame has the expected columns
- numeric dtypes are preserved
- row count matches the number of MST edges

Why it matters:

This is a small helper, but it affects usability in notebooks, reports, and diagnostics.

### 6.10 Metric-specific behavior

These tests validate the metric branches inside `build_core_sg_from_data(...)`.

What is tested:

- standard metrics such as `euclidean` run correctly
- `minkowski` uses the configured `p`
- `arccos` maps to cosine pairwise distances as implemented
- the code path remains deterministic for the same input

Why it matters:

The graph structure depends directly on the pairwise distance computation, so metric-specific regressions can silently affect all downstream results.

### 6.11 Precomputed-style and missing-raw-data behavior

These tests validate behavior when raw feature data is unavailable for wrapped MST visualization.

What is tested:

- MST wrapper access returns `None` with a warning when raw data is unavailable
- raw arrays remain available even when the wrapped plotting object cannot be built

### 6.12 Responsibility grouping and naming conventions

The current suite also enforces organizational conventions intended to keep the repository maintainable over time.

What is expected:

- unit, integration, and validation tests are placed in separate directories
- files use consistent `test_<subject>_<behavior>.py` naming
- classes use `Test...` grouping names
- slower algorithm-validation tests are easy to identify and run separately

Why it matters:

Good test structure reduces contributor confusion, makes future gaps easier to spot, and lowers the cost of maintaining the suite as the project evolves.

Why it matters:

This behavior is intentional in the current implementation and should be documented and tested so users understand the limitation.

## 7. Notes about documentation quality

This document is intentionally written in one language only and uses repository paths that match the current package metadata:

- package import: `core_sg`
- main implementation: `core_sg/core_sg.py`
- editable installation: `pip install -e ".[test]"` or `pip install -e ".[dev]"`

When updating test documentation in the future:

- avoid mixing Portuguese and English in the same guide
- keep command examples copy-paste ready
- reference real package paths only
- prefer `pytest` examples over manual script execution instructions
- update this file whenever the public API or test layout changes

## 8. Summary

In practice, the current test documentation should help answer three questions quickly:

1. How do I install the dependencies needed to validate the project locally?
2. How do I run the automated checks?
3. What behavior is each test group supposed to validate?

For Core-SG, the most important coverage areas are:

- correct support construction at `k_max`
- safe reuse for smaller `k`
- valid MST extraction
- correct hierarchy reconstruction
- explicit handling of invalid inputs
- stable exposure of wrapped HDBSCAN-style objects
