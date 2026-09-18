from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

extensions = [
    Extension(
        "core_sg._mst_kruskal",
        sources=["core_sg/_mst_kruskal.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
    ),
    Extension(
        "core_sg._reweight",
        sources=["core_sg/_reweight.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    package_data={
        "core_sg": ["*.pyd", "*.py", "*.pyx"],
    },
)