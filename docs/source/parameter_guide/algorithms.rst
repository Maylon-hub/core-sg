Algorithms
==========

``algorithm`` selects how the internal support graph is built. Most users
configure it on :class:`core_sg.CoreSGClusterer`:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30, algorithm="core-sg")

Available Algorithms
--------------------

``"core-sg"``
   Default exact Core-SG construction path.

``"score-sg"``
   Approximate anti-hub reinforced construction path based on PyNNDescent.

``algorithm="core-sg"``
-----------------------

This is the default and most conservative option.

The exact Core-SG path builds pairwise distance information, constructs the
support graph at ``k_max``, and then lets later ``fit(X, k=...)`` calls reuse
that support through the estimator's internal ``core_sg_`` object.

Use ``"core-sg"`` when:

* you want the default HDBSCAN-style Core-SG behavior;
* you prefer exact graph construction over approximate neighbor discovery;
* your dataset size makes dense pairwise distance construction acceptable;
* reproducibility and reference-style behavior are more important than avoiding
  the dense distance matrix.

Tradeoffs:

* can be more expensive up front;
* stores dense pairwise distance information internally;
* is usually the safest first choice.

Example:

.. code-block:: python

   clusterer = CoreSGClusterer(
       k_max=30,
       algorithm="core-sg",
       metric="euclidean",
   )
   clusterer.fit(X, k=20)

``algorithm="score-sg"``
------------------------

Score-SG is the approximate variant. It uses PyNNDescent to build an
approximate nearest-neighbor graph, derives approximate core-distance
information, selects anti-hubs by directed in-degree, adds anti-hub support
edges, and then reuses the same downstream MST and hierarchy pipeline.

Use ``"score-sg"`` when:

* dense all-pairs distance construction is too expensive;
* an approximate sparse-neighbor workflow is acceptable;
* you want to experiment with anti-hub reinforced support graphs;
* you can validate that results are stable enough for your analysis.

Tradeoffs:

* neighbor membership is approximate;
* ``random_state`` matters for tied anti-hub selection;
* ``approx_knn_kwargs`` can affect both runtime and output quality;
* the support graph may be disconnected, in which case MST extraction is not
  possible and Core-SG raises an explicit error.

Example:

.. code-block:: python

   clusterer = CoreSGClusterer(
       k_max=30,
       algorithm="score-sg",
       metric="euclidean",
       random_state=42,
       approx_knn_kwargs={"n_trees": 8},
   )
   clusterer.fit(X, k=20)

Related Parameters
------------------

``random_state``
   Used by Score-SG for reproducible random tie-breaking during anti-hub
   selection.

``approx_knn_kwargs``
   Optional dictionary forwarded to PyNNDescent when ``algorithm="score-sg"``.

``metric`` and ``p``
   Affect distance computation and neighbor structure in both algorithms.

Practical Recommendation
------------------------

Start with ``algorithm="core-sg"`` unless you specifically need the approximate
Score-SG path. Move to ``algorithm="score-sg"`` when the dense exact path is
too costly or when the anti-hub construction is part of the experiment.
