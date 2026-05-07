Limitations
===========

Performance results depend on:

* sample size;
* feature dimension;
* metric choice;
* selected ``k_max`` and target ``k`` values;
* HDBSCAN version and private API behavior;
* whether Cython extensions are available;
* PyNNDescent warm-up and approximate neighbor settings.

The benchmark pages should not be read as universal guarantees. They document
the intended repeated multi-``k`` interpretation and provide reproducible
scripts and figures for the repository's benchmark data.
