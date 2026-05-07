Scaling
=======

.. image:: ../_static/images/benchmarks/coresg_runtime_breakdown.png
   :alt: Core-SG runtime breakdown

Core-SG scaling depends on the cost of support graph construction and the cost
of repeated MST extraction.

The build stage typically dominates cumulative Core-SG cost. The value of the
method comes from avoiding repeated full builds for every target ``k``.

Score-SG is designed to avoid explicit dense all-pairs distance construction,
but it should still be interpreted empirically because approximate nearest
neighbor behavior depends on data geometry, metric choice, and PyNNDescent
parameters.
