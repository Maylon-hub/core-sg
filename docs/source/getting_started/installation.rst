Installation
============

This RC supports CPython ``3.11`` within the Windows/Linux x86-64 platform
scope. Windows is locally qualified; Linux qualification must pass GitHub
Actions before publication. Linux wheels target glibc ``>=2.28``. macOS is
not currently qualified and remains future work, without an incompatibility
claim. Other Python versions are not advertised by this RC.

Runtime dependencies are declared in
``pyproject.toml`` and include NumPy, pandas, scikit-learn, HDBSCAN, and
PyNNDescent.

Install From PyPI
-----------------

.. code-block:: bash

   pip install core-sg-mustache==0.4.5rc3

This candidate is not published yet. For local qualification install the built
wheel instead. Do not co-install the separate ``core-sg`` distribution because
both use the ``core_sg`` import namespace. The linked MIDAS site describes upstream.

Local Development Install
-------------------------

From the repository root:

.. code-block:: bash

   pip install -e .

Install test and contributor tooling:

.. code-block:: bash

   pip install -e ".[dev]"

Install documentation tooling:

.. code-block:: bash

   pip install -e ".[docs]"

Build the documentation locally:

.. code-block:: bash

   python docs/scripts/generate_diagrams.py
   python docs/scripts/generate_benchmark_figures.py
   sphinx-build -b html docs/source docs/build/html

Use the strict command before opening documentation pull requests:

.. code-block:: bash

   sphinx-build -W -b html docs/source docs/build/html

Dependency Notes
----------------

The ``docs`` extra includes Sphinx, the Read the Docs theme, autodoc helpers,
MyST, nbsphinx, IPython, matplotlib, pandas, and scikit-learn. It intentionally
keeps notebook execution disabled during ordinary documentation builds so the
site remains fast and deterministic.
