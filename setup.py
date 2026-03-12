from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent


def readme() -> str:
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Core-SG graph construction and MST extraction utilities for HDBSCAN-style clustering."


def requirements() -> list[str]:
    req_path = BASE_DIR / "requirements.txt"
    if not req_path.exists():
        return []
    return [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def dev_requirements() -> list[str]:
    return [
        "pytest>=8.0",
        "pytest-cov>=5.0",
        "build>=1.2.2",
        "twine>=5.1.0",
    ]


configuration = {
    "name": "core-sg",
    "version": "0.1.0rc1",
    "description": "Core-SG graph construction and MST extraction utilities for HDBSCAN-style clustering.",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "author": "Midas Core-SG Team",
    "author_email": "gmcorlando@estudante.ufscar.br",
    "url": "https://github.com/midas-core-sg/core-sg",
    "project_urls": {
        "Source": "https://github.com/midas-core-sg/core-sg",
        "Issues": "https://github.com/midas-core-sg/core-sg/issues",
        "Documentation": "https://github.com/midas-core-sg/core-sg#readme",
    },
    "keywords": [
        "core-sg",
        "clustering",
        "hdbscan",
        "graph",
        "mst",
        "density-based-clustering",
    ],
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "license": "BSD-3-Clause",
    "license_files": ["LICENSE"],
    "packages": find_packages(include=["core_sg", "core_sg.*"]),
    "include_package_data": True,
    "install_requires": requirements(),
    "extras_require": {
        "dev": dev_requirements(),
        "test": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
        ],
    },
    "python_requires": ">=3.10",
    "zip_safe": False,
}

setup(**configuration)