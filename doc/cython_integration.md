# Cython Integration Notes

This document explains how Cython is currently used in Core-SG, how the build is integrated into the package, and which design decisions were made to keep the public API stable while accelerating the most expensive paths.

Its purpose is to give contributors a single technical reference for the Cython-backed implementation instead of spreading that information across code comments, commits, and benchmarks.

## 1. Why Cython was introduced

The project has two hot paths with different performance characteristics:

- `kruskal_mst(...)` is dominated by Python-level loops and repeated Union-Find operations.
- `reweight_core_sg_mutual_reachability(...)` is mostly NumPy-based, but still spends time in repeated lookup preparation and in the remaining per-edge work.

These two cases led to two different but compatible ideas:

1. move the tight Kruskal loop and Union-Find operations into Cython
2. keep the public reweight function in Python, but move the numeric kernel to Cython after removing repeated preprocessing

The implementation keeps the public API unchanged. Function names, signatures, return contracts, sorting behavior, and error behavior remain compatible with the Python version.

## 2. Which modules are compiled

The Cython-backed code lives in two internal modules:

- `core_sg/_mst_kruskal.pyx`
- `core_sg/_reweight.pyx`

These are internal implementation modules. They are not meant to be public entry points.

The public functions still live in the Python modules:

- `core_sg/mst_kruskal.py`
- `core_sg/reweight.py`

Those Python modules act as compatibility-preserving wrappers. They validate inputs, preserve the public interface, and dispatch to the compiled backend when it is available.

## 3. How the public interface is preserved

The public API was intentionally kept in Python for three reasons:

1. input validation and error messages remain easy to audit
2. import paths do not change
3. a pure-Python fallback still exists if the compiled extension is unavailable

In practice, the wrapper pattern is:

- validate the same input constraints as before
- call the compiled implementation if import succeeds
- otherwise fall back to a Python reference implementation

This design makes behavior easier to compare during tests because the previous algorithm still exists in a readable form inside the package.

## 4. `kruskal_mst(...)` design

The Kruskal acceleration targets the parts that were previously expensive in Python:

- Union-Find state updates
- path halving in `find`
- union by rank in `union`
- the main loop that consumes sorted edges and writes MST edges

The Python wrapper still performs the high-level contract work:

- checks the edge array shape
- checks `n_nodes`
- keeps the same `np.recarray` return contract

The compiled function receives already-normalized arrays and emits the same structured output shape and field names:

- `u`
- `v`
- `distance`

Tie-breaking behavior is preserved by keeping the same `np.lexsort((u, v, w))` ordering logic before the main loop.

## 5. `reweight_core_sg_mutual_reachability(...)` design

The reweight path was not treated as a direct line-by-line Cython rewrite.

Instead, the implementation was split into two layers:

### 5.1 Python-side preprocessing and cache

The Python wrapper now preprocesses `metric_edges` into a sorted lookup only once per `(id(metric_edges), n_nodes)` pair.

That cache stores:

- sorted integer keys
- sorted edge weights
- a weak reference to the original `metric_edges` array

The weak reference prevents the cache from incorrectly reusing stale entries when the original array object no longer exists.

This preprocessing step matters because repeated multi-`k` workflows reuse the same support graph and `metric_edges`, so rebuilding the lookup every call wastes time.

### 5.2 Cython-side numeric kernel

After the cache is prepared, the Cython kernel performs the per-edge work:

- compute the normalized undirected edge key
- binary-search the prepared lookup
- raise `KeyError` if an edge is missing
- compute `max(core_k[u], core_k[v], dist_uv)`
- write the final weight back into the edge array

The final sort remains in Python through `sort_core_sg(...)`, which keeps the ordering contract explicit and easy to compare with the original implementation.

## 6. Build and packaging integration

The packaging configuration is intentionally centralized in `pyproject.toml`.

### 6.1 Build requirements

The build system declares:

- `setuptools`
- `wheel`
- `cython`
- `numpy`

These are needed because the project builds Cython extensions as part of package installation and wheel creation.

### 6.2 Extension declaration

The extensions are declared in `pyproject.toml` under `tool.setuptools.ext-modules`.

This means:

- there is no `setup.py`
- editable installs can build the extensions directly from project metadata
- `python -m build --no-isolation` can build both `sdist` and wheel using the same configuration

### 6.3 Why the `.pyx` files avoid `numpy.get_include()`

One practical challenge of removing `setup.py` is that `numpy.get_include()` is usually called from Python build code.

To avoid that dependency on a custom build script, the Cython modules were written so they do not require NumPy C headers through `cimport numpy`.

Instead, they use:

- standard `import numpy as np` at the Python level
- typed memoryviews (`long long[:]`, `double[:]`, `double[:, :]`) for the compiled loops
- NumPy arrays created from Python space and then viewed through memoryviews

This keeps the compiled code efficient enough for the targeted loops while allowing the extensions to be declared directly in `pyproject.toml`.

## 7. Generated artifacts and repository hygiene

Cython generates build artifacts during local compilation, especially:

- generated C files
- compiled extension binaries such as `.so`
- standard build output directories

These artifacts are intentionally ignored in `.gitignore` so local builds do not pollute the worktree.

The source distribution still includes the `.pyx` sources and generated `.c` files through `MANIFEST.in`, which keeps packaging reproducible and explicit.

## 8. Validation strategy used during the migration

The migration was validated incrementally rather than only at the end.

The checkpoints used were:

1. baseline targeted tests and benchmark measurements
2. Kruskal migration validation
3. reweight cache refactor validation
4. compiled reweight backend validation
5. package build validation

The test coverage was extended with focused checks for:

- public signature preservation
- equivalence between Python and compiled backends
- tie cases for Kruskal ordering stability
- disconnected graph error behavior
- repeated multi-`k` cache reuse

The package was also validated with:

- editable installation
- `python -m build --no-isolation`
- unit, integration, and validation test runs

## 9. Current tradeoffs

The current design intentionally favors maintainability over maximum compiler-specific optimization.

Examples of those tradeoffs:

- public wrappers remain in Python for clarity and compatibility
- `sort_core_sg(...)` remains in Python/NumPy
- the reweight cache is simple and local rather than introducing a more complex cache manager
- the implementation uses typed memoryviews instead of a deeper NumPy C-API integration

This keeps the codebase easier to review and reduces packaging complexity while still delivering clear improvements on the hot paths.

## 10. Practical guidance for contributors

If you change the Cython-backed paths, follow this sequence:

1. preserve the Python wrapper signature and exceptions
2. preserve sorting and tie-breaking behavior
3. keep the Python fallback implementation usable for equivalence checks
4. rerun targeted tests first
5. rerun package build validation after the code change

Recommended local commands:

```bash
pip install --no-build-isolation -e ".[dev]"
pytest tests/unit/test_graph_helpers.py -q
pytest tests/integration/test_reference_equivalence.py -q
pytest tests/validation -q
python -m build --no-isolation
python benchmarks/benchmark_graph_ops.py
```

## 11. Important note about setuptools support

The current `setuptools` support for `tool.setuptools.ext-modules` in `pyproject.toml` works for this project, but it is still marked as experimental by setuptools itself.

That does not block the build, but contributors should be aware of it when upgrading the packaging toolchain.
