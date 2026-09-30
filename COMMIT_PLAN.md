# Proposed commit plan — not executed

CORE-SG branch `develop`, base HEAD `f5a716b202ced04ef3bf7a9abc9e1f12697cb4f1`.

This plan includes existing uncommitted corrections and this qualification task.
No staging, commit, push, tag or history rewrite has been performed. Review the
complete diff before any authorized staging; do not discard user changes.
Each file is assigned once. Some files contain previously audited fixes together
with release edits; split hunks only if the owner requests a finer history.
Run mandatory CI on the FINAL commit of the complete series, not intermediate
commits with intentionally incomplete version/dependency alignment.

## 1. fix: preserve complete CORE-SG support across hierarchy extraction

- `core_sg/core_sg.py`
- `core_sg/hdbscan_adapter.py`

## 2. test: cover private API, Euclidean reference and release gates

- `tests/integration/test_private_hdbscan_contract.py`
- `tests/test_release_promotion.py`
- `tests/unit/test_hdbscan_adapter.py`
- `tests/unit/test_public_euclidean_reference.py`

## 3. build: qualify Windows/Linux cp311 native artifacts

- `.github/workflows/ci.yml`
- `.github/workflows/docs.yml`
- `.github/workflows/release.yml`
- `.github/workflows/test.yml`
- `.github/workflows/wheels.yml`
- `MANIFEST.in`
- `core_sg/__init__.py`
- `pyproject.toml`
- `requirements/README.md`
- `requirements/build-win-py311.txt`
- `requirements/runtime-win-py311.txt`
- `scripts/build_rc_artifacts.py`
- `scripts/check_artifacts.py`
- `scripts/collect_release_artifacts.py`
- `scripts/qualify_artifacts.py`
- `scripts/rc_smoke.py`
- `scripts/run_installed_tests.py`
- `scripts/validate_promotion.py`
- `setup.py`

## 4. docs: record attribution, platform policy and RC readiness

- `AUTHORS.md`
- `AUTHORSHIP_APPROVAL_REQUIRED.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `COMMIT_PLAN.md`
- `DATASETS.md`
- `README.md`
- `RELEASE_NOTES.md`
- `SECURITY_AUDIT_2026-09-27.md`
- `docs/source/conf.py`
- `docs/source/getting_started/installation.rst`
- `testpypi_readiness_2026-09-27.md`

## Coordination before any remote write

Obtain owner approval of repository manuscript/privacy/license findings in
`SECURITY_AUDIT_2026-09-27.md`. Publication additionally needs authorship approval.
After explicit commit/push authorization, prepare CORE-SG first, record its final
immutable SHA, and qualify MustaCHE against that SHA on Windows/Linux. Record
both successful CI run IDs; create version tags only after separate authorization.
MacOS qualification is outside this RC; do not expand Python/platform claims
from workflow configuration alone. Recheck this inventory if any file changes.
