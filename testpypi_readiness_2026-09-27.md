# TestPyPI readiness — paired RC — 2026-09-27

Canonical combined report: the companion MustaCHE repository's
`testpypi_readiness_2026-09-27.md`, covering mustache-core 0.3.0rc3 and
core-sg-mustache 0.4.5rc3. It contains Windows results, artifact hashes,
qualification/promotion gates and exact human checklists. It is not yet a
remote published document; consult the local paired checkout.

Official RC targets: **Windows x86-64 and Linux x86-64, CPython 3.11 only**.
Linux requires real GitHub Actions evidence after authorized commit/push.
**macOS = NOT QUALIFIED / FUTURE WORK**, not a TestPyPI blocker.

CORE-SG version: **0.4.5rc3**. Native HDBSCAN: **0.8.44**, pinned. Distribution
name is `core-sg-mustache`, import namespace `core_sg`; never co-install the
separate `core-sg` distribution into the same environment.

- Windows wheel: installed native imports/smoke/pip check and **222 full-suite
  tests passed**. Source archive installation/import/pip check/smoke and its
  **222 full-suite tests passed**; details are in the combined report.
- Linux: no revised remote run or native artifact yet; no invented compatibility
  claim. CI builds a common sdist, repaired cp311-manylinux x86-64 wheel and
  cp311-win_amd64 wheel, then qualifies fresh wheel/sdist installations.
- cibuildwheel 3.3.1 identifiers verified; local NuGet interpreter-download
  attempt stalled. The qualified Windows wheel was built from the same sdist
  using the local compiler, not claimed as a cibuildwheel pass.
- CI validation and manual TestPyPI/PyPI promotion are separated. Promotion
  requires both OS reports, full suites, matching final source SHA and hashes.
- TestPyPI publisher: repository `Maylon-hub/core-sg`, workflow `wheels.yml`,
  environment `testpypi`. Account/ownership/approver state requires human
  confirmation; no credentials were accessed or changed.
- Matching `v0.4.5rc3` tag and registry version were not found in fresh checks;
  this is not a version reservation. No tag was created.
- CFF schema and release files are valid; software attribution needs advisor/
  MIDAS/original maintainer approval in `AUTHORSHIP_APPROVAL_REQUIRED.md`.
- Repository/history privacy/license review is described separately in
  `SECURITY_AUDIT_2026-09-27.md`. Proposed file groups are in `COMMIT_PLAN.md`.

Ready for commit: **YES** — technical preparation, authorization required.

Ready for remote CI: **YES** — authorized push/run still required.

Ready for TestPyPI artifacts: **NO** — actual Linux/native remote qualification
and Windows/Linux cibuildwheel CI evidence missing.

Ready for TestPyPI publication: **BLOCKED** — authorize commit/push/CI, obtain
passing required jobs and authorship/sharing approval, verify Trusted Publisher
and environments, then separately authorize matching tags and publication.

No staging, commit, push, tag, upload, repository transfer, RNG work or Git
history rewrite was performed in this task. Earlier dirty work is preserved.
