Metrics
=======

``metric`` controls distance computation during graph construction. The default
is ``"euclidean"``.

``p`` is used for distance families such as Minkowski:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30, metric="minkowski", p=2)

The default exact ``algorithm="core-sg"`` path builds dense pairwise distance
information internally. The approximate ``algorithm="score-sg"`` path uses
PyNNDescent for neighbor discovery and computes exact distances only for
selected support edges.

Metric choice affects nearest-neighbor structure, core distances,
mutual-reachability weights, MST extraction, and final hierarchy outputs.
