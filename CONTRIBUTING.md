# Contributing to core-sg

Thank you for your interest in contributing to `core-sg`.

This document describes the recommended development, branching, testing, and release workflow for the project.

---

## Development workflow

All new work should follow the branch structure below.

### Main branches

#### `main`
Stable branch of the project.

This branch should reflect only released or release-ready versions of the library.

Direct development on `main` is not recommended.

#### `develop`
Integration branch for the next version.

Everything that is planned for the next release should go through `develop` first.

This is also the branch from which pre-release and final release tags are created.

### Feature branches

All new work should be developed in isolated feature branches created from `develop`.

Branch naming convention:

- `feat/...` for new features
- `fix/...` for bug fixes
- `chore/...` for maintenance tasks
- `docs/...` for documentation updates

Examples:

- `feat/class-core-sg`
- `feat/test-suite`
- `fix/mst-ordering`
- `docs/readme-update`

---

## Standard contribution flow

### 1. Create a branch from `develop`

```
git checkout develop
git pull origin develop
git checkout -b feat/my-feature
```

### 2. Work locally

Make small, descriptive commits during development.

Example:

```
git add .
git commit -m "feat: add hierarchy extraction to CoreSG"
```

### 3. Run tests locally

Before opening a pull request, make sure the test suite passes.

```
pytest tests -v -ra
```

### 4. Open a Pull Request to `develop`

All feature branches must be merged into `develop` through a Pull Request.

Expected flow:

- push your branch
- open a PR to `develop`
- wait for CI to run
- request review if needed
- merge after approval and passing checks

In short:

[IDENT]feature branch -> develop[IDENT]

---

## What should go into `develop`

The `develop` branch should contain the integrated state of the next release.

This includes:

- new features
- bug fixes
- tests
- packaging adjustments
- documentation
- CI workflow changes

Do not use `develop` for direct unreviewed work unless absolutely necessary.

---

## Pre-release workflow

Pre-releases are used to validate the package on TestPyPI before publishing a final version to PyPI.

### When to create a pre-release

Create a pre-release when `develop` is mature enough to test:

- package build
- metadata
- installation
- importability
- minimal runtime behavior

### Versioning for pre-releases

Use versions such as:

- `0.1.0rc1`
- `0.1.0rc2`

### Preparing a pre-release

Update the package version on `develop`, commit the change, and push it.

```
git checkout develop
git pull origin develop
```

Then create and push the tag:

```
git tag -a v0.1.0rc1 -m "core-sg v0.1.0rc1"
git push origin v0.1.0rc1
```

git tag -a v0.1.0rc3 -m "core-sg v0.1.0rc3"
git push origin v0.1.0rc3

### What happens next

Once the tag is pushed:

- the package build workflow runs
- the TestPyPI publishing workflow runs
- the package is published automatically to TestPyPI

### If something needs to be fixed

If the pre-release needs corrections:

1. go back to `develop`
2. apply the fixes
3. bump the version to the next pre-release, for example `0.1.0rc2`
4. commit
5. create a new tag
6. publish again to TestPyPI

Do not reuse the same pre-release version after making changes.

---

## Final release workflow

A final release should only happen after a pre-release has been validated.

### Final version format

Examples:

- `0.1.0`
- `0.1.1`
- `0.2.0`

### Preparing the final release

Update the version on `develop`, commit, and push:

```
git checkout develop
git pull origin develop
git add .
git commit -m "chore(release): prepare v0.1.0"
git push origin develop
```

Create and push the final tag:

```
git tag -a v0.1.0 -m "core-sg v0.1.0"
git push origin v0.1.0
```

### What happens next

Once the final tag is pushed:

- the package build workflow runs
- the PyPI publishing workflow runs
- the package is published automatically to PyPI

---

## Synchronizing `develop` and `main`

After a final release is published, `main` should receive the stable release state.

Recommended flow:

[IDENT]develop -> main[IDENT]

### Procedure

1. open a Pull Request from `develop` to `main`
2. review and merge it after checks pass

This ensures:

- `main` reflects the stable released state
- `develop` remains the integration branch for the next version

---

## Versioning policy

The project should follow semantic versioning.

Format:

[IDENT]MAJOR.MINOR.PATCH[IDENT]

Examples:

- `0.1.0`
- `0.1.1`
- `0.2.0`
- `1.0.0`

### Pre-releases

Use release candidates:

- `0.1.0rc1`
- `0.1.0rc2`

### General rules

- PATCH for backward-compatible fixes
- MINOR for backward-compatible new functionality
- MAJOR for breaking changes

As long as the API is still evolving, it is natural to remain in the `0.x.y` range.

---

## Running tests

Use the standard test command:

```
pytest tests -v -ra
```

The repository CI should also run the test workflow automatically on relevant pushes and pull requests.

---

## Coding expectations

Contributions should aim to keep the project:

- clear
- reproducible
- testable
- consistent with the existing codebase

When possible:

- add tests for new behavior
- avoid unrelated changes in the same PR
- keep pull requests focused
- document public API changes

---

## What to avoid

Please avoid:

- developing directly on `main`
- creating release tags from feature branches
- publishing from feature branches
- reusing the same release version after modifying the code
- mixing unrelated features in a single PR

---

## Recommended project flow

The standard project flow should be:

- feat/* -> develop -> vX.Y.ZrcN -> TestPyPI -> vX.Y.Z -> PyPI -> main[IDENT]

This keeps development, integration, release validation, and stable publication clearly separated.

---

## HDBSCAN acknowledgment

`core-sg` is structurally inspired by and technically dependent on the `hdbscan` library.

If your contribution touches areas that reuse or adapt ideas or implementation paths from `hdbscan`, please preserve that relationship clearly in documentation and attribution where appropriate.

---

## Questions

If you are unsure about the correct contribution path, prefer opening your work against `develop` and keeping the change focused and reviewable.