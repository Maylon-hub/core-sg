# Changelog

## 0.4.5 — Stable candidate (unpublished)

- Carry forward the scientifically and functionally qualified `0.4.5rc3`
  implementation without changing graph construction or extraction algorithms.
- Reuse one complete `k_max` support graph for multiple smaller `k` MSTs and
  hierarchies; retain the corrected reference-distance and native-tree behavior.
- Keep the pinned HDBSCAN private-API compatibility contract and native
  Cython extensions. Build and qualify wheel/sdist on Windows and Linux x86-64,
  CPython 3.11 only; macOS remains unqualified.
- Correct stable version, citation and public Changelog metadata. The previous
  TestPyPI RC and its tag remain immutable.

Known limitations: dense exact distances require quadratic memory;
approximate neighbor graphs may disconnect; no unseen-point prediction;
HDBSCAN private APIs remain pinned to `hdbscan==0.8.44`. No RNG equivalence
has been demonstrated.

## 0.4.5rc3 — TestPyPI pre-release

- Narrow official RC scope to Windows/Linux x86-64, CPython 3.11;
  produce only win_amd64/manylinux x86-64 wheels. Require fresh wheel/sdist
  native and full-suite qualification; macOS is unqualified future work.

- Preserve the complete `k_max` support after lower-k extraction and reuse it.
- Correct public Euclidean distance/neighbor conventions and the seed tree's
  reference-compatible behavior; test non-rounded data against native HDBSCAN.
- Respect `no_noise=False`; retain the native-tree compatibility adapter.
- Add support-reuse, public distance-matrix and MST-weight regressions.
- Qualify native private APIs against pinned `hdbscan==0.8.44`.
- Align fork metadata/citations; build extensions instead of collecting stale
  interpreter-specific `.pyd` files; separate validation/manual publishing.
- Exclude generated C with machine-specific paths; regenerate it from packaged
  `.pyx` sources in the isolated build.

These changes do not demonstrate algorithmic equivalence to RNG.
See RELEASE_NOTES.md for scope and compatibility.

## Earlier versions

Historical tags, including 0.4.4, 0.4.5rc1 and 0.4.5rc2, remain unchanged.
This consolidation is not retroactively attributed to their artifacts.
