from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestGraphHelpers:
    def test_knn_from_precomputed_excludes_self_and_returns_sorted_neighbors(
        self, core_sg_module
    ):
        D = np.array(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 3.0],
                [1.0, 3.0, 0.0],
            ]
        )

        idx, dist = core_sg_module.knn_from_precomputed(D, k=2, include_self=False)

        assert idx.shape == (3, 2)
        assert dist.shape == (3, 2)
        assert np.array_equal(idx[0], np.array([2, 1]))
        assert np.all(dist[:, 0] <= dist[:, 1])

    def test_build_knng_vectors_preserves_shape_and_distances(self, core_sg_module):
        idxs = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        dists = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 3.0]], dtype=np.float64)

        metric_edges, knng = core_sg_module.build_knng_vectors(
            idxs, dists, knng_size=3, k_max=2
        )

        assert metric_edges.shape == (6, 3)
        assert knng.shape == (6, 3)
        assert np.array_equal(knng[0], np.array([0.0, 1.0, 1.0]))
        assert np.array_equal(metric_edges[0], np.array([1.0, 0.0, 1.0]))

    def test_reweight_core_sg_mutual_reachability_uses_max_rule(self, core_sg_module):
        core_sg = np.array([[0, 1, -1.0], [1, 2, -1.0]], dtype=np.float64)
        metric_edges = np.array([[1, 0, 1.0], [2, 1, 4.0]], dtype=np.float64)
        core_k = np.array([0.5, 2.0, 3.0], dtype=np.float64)

        weighted = core_sg_module.reweight_core_sg_mutual_reachability(
            core_sg=core_sg,
            core_k=core_k,
            metric_edges=metric_edges,
            n_nodes=3,
        )

        assert np.array_equal(weighted[:, :2], np.array([[0.0, 1.0], [1.0, 2.0]]))
        assert np.allclose(weighted[:, 2], np.array([2.0, 4.0]))

    def test_kruskal_mst_builds_tree_with_n_minus_one_edges(self, core_sg_module):
        edges = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 2.0],
                [0, 2, 10.0],
                [2, 3, 1.0],
            ],
            dtype=np.float64,
        )

        mst = core_sg_module.kruskal_mst(edges, n_nodes=4)

        assert mst.shape == (3,)
        assert np.array_equal(mst.u, np.array([0, 2, 1]))
        assert np.array_equal(mst.v, np.array([1, 3, 2]))
        assert np.allclose(mst.distance, np.array([1.0, 1.0, 2.0]))

    def test_knn_from_precomputed_rejects_non_square_distance_matrix(
        self, core_sg_module
    ):
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="NxN"):
            core_sg_module.knn_from_precomputed(D, k=1)
