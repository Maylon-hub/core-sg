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


configuration = {
    "name": "core-sg",
    "version": "0.1.0",
    "description": "Core-SG graph construction and MST extraction utilities for HDBSCAN-style clustering.",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    "keywords": "core-sg clustering hdbscan graph mst density hierarchical",
    "url": "https://github.com/midas-core-sg/core-sg",
    "author": "Midas Core-SG Team",
    "license": "-",
    "packages": find_packages(include=["core_sg", "core_sg.*"]),
    "install_requires": requirements(),
    "python_requires": ">=3.10",
    "include_package_data": True,
    "zip_safe": False,
}

setup(**configuration)