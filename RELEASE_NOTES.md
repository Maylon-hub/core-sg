# CORE-SG MustaCHE integration fork 0.4.5rc3

Unpublished RC proposal; companion of `mustache-core==0.3.0rc3`.

For this personal-fork RC, the software citation retains the existing Midas
Core-SG Team authorship and Gabriel Orlando's implementation credit. Maylon
Martins de Melo is credited as a contributor to integration, support reuse,
tests and packaging, not added as a CORE-SG software author. The scientific
paper remains a separate reference.

## Official RC platform policy

Windows x86-64 and Linux x86-64 (glibc >= 2.28), **CPython 3.11 only**.
Publish only cp311 Windows AMD64 and manylinux x86-64 native wheels, plus sdist.
Windows has local evidence; Linux requires a real GitHub Actions pass.
macOS is NOT QUALIFIED / FUTURE WORK and is not required for this RC. This does
not claim macOS incompatibility. Python 3.10/3.12/3.13 require later qualification.

## Scientific scope

Build support at `k_max`, extract smaller-k MSTs/hierarchies without rebuilding.
The implementation uses HDBSCAN components. Targeted reference tests are not
universal proof of equality to RNG, every HDBSCAN tie case, or the original
paper's implementation. SCORE-SG remains a separate approximate path.

## Known limitations

Exact dense distances require quadratic memory. Approximate neighbor graphs
may be disconnected. Prediction for unseen points is not implemented.
Private `hdbscan.hdbscan_._tree_to_labels`, linkage and plotting APIs may break
upstream: this RC pins `hdbscan==0.8.44`. Scikit-learn HDBSCAN is not the owner
of that private API. Future upgrades require real-dependency compatibility tests.
Local qualification does not automatically validate Linux/macOS.

## Backward compatibility

`core_sg`, `CoreSG`, `CoreSGClusterer` and extraction signatures remain.
The distribution is **core-sg-mustache**, not the separate `core-sg` package;
do not install both into one environment (shared import namespace).
New `__version__` reports installed metadata. Synthetic tests are distributed;
raw datasets and benchmark outputs are excluded from wheel/sdist.

## Breaking changes

No intentional signature removal. The native HDBSCAN dependency is now exact.
Python metadata is deliberately narrowed to >=3.11,<3.12 for this RC.
Corrected support/distance handling can change previously incorrect outputs.
Historical benchmark numbers are not fresh RC performance claims.
