# CORE-SG MustaCHE integration fork 0.4.5

Stable `core-sg-mustache==0.4.5` is published on TestPyPI and paired with
`mustache-core==0.3.0`. Official PyPI publication is deferred. The historical
`0.4.5rc3` TestPyPI release remains an immutable pre-release record. The stable
release changes version/release metadata relative to rc3, not scientific
algorithms.

CORE-SG constructs one complete support graph at `k_max` and reuses it to
extract MSTs and hierarchies for smaller `k`. Native Cython Kruskal and
reweighting extensions ship in the wheels. Corrections to support reuse,
distance/reference compatibility and packaging were tested in the RC and
requalified in the final stable artifacts. SCORE-SG remains a separate
approximate path.

## Supported platforms

Windows x86-64 and Linux x86-64 (glibc >= 2.28), **CPython 3.11 only**.
The stable Windows/Linux wheel and sdist paths passed qualification; an RC
pass alone would not be final-artifact evidence. macOS and other Python versions are
not qualified, not declared incompatible.

## Known limitations

Exact dense distances use quadratic memory. Approximate neighbor graphs may
disconnect. Prediction for unseen points is not implemented. The private
`hdbscan.hdbscan_._tree_to_labels` contract is pinned to `hdbscan==0.8.44`;
future upgrades require compatibility tests. Targeted reference tests are not
proof of equivalence to RNG, every HDBSCAN tie case or the original paper's
implementation. Historical benchmarks are not fresh stable-release results.

## Backward compatibility and breaking changes

`core_sg`, `CoreSG`, `CoreSGClusterer` and extraction signatures remain. No
new signature removal is intended relative to rc3. The distribution is
`core-sg-mustache`, not the separate `core-sg` package; do not co-install both
because they share the import namespace. The qualified Python restriction
`>=3.11,<3.12` and exact `hdbscan==0.8.44` pin remain. Corrections already
present in rc3 may change outputs relative to older, incorrect versions.

## Attribution

The software citation retains the Midas Core-SG Team authorship and Gabriel
Orlando's implementation credit. Maylon Martins de Melo is credited in
`AUTHORS.md` as a contributor to integration, support reuse, tests and
packaging, not added as a CORE-SG software author. The method paper remains
a separate reference.
