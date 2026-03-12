from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import make_fit_payload


@pytest.fixture()
def fitted_obj(core_sg_module, monkeypatch, sample_X):
    payload = make_fit_payload()

    def fake_build_core_sg_from_data(X, k_max, metric, p, test_only=False):
        return payload

    monkeypatch.setattr(core_sg_module, "build_core_sg_from_data", fake_build_core_sg_from_data)
    obj = core_sg_module.CoreSG(cluster_selection_method="leaf", allow_single_cluster=True)
    obj.fit(sample_X, 4, test_only=True)
    return obj, payload


def test_get_core_distance_returns_requested_column(fitted_obj):
    obj, payload = fitted_obj
    expected = payload[2][:, 1]

    result = obj.get_core_distance(2)

    assert np.array_equal(result, expected)


def test_get_core_sg_mutual_reachability_distance_delegates_to_helper(core_sg_module, fitted_obj, monkeypatch):
    obj, _ = fitted_obj
    sentinel = np.array([[0.0, 1.0, 3.14]])

    def fake_core_sg_mutual_reachability_distance(core_sg, metric_edges, core_k_list, n_nodes,k_max, k):
        assert core_sg is obj._core_sg
        assert metric_edges is obj._metric_edges
        assert core_k_list is obj._core_k_list
        assert n_nodes == obj.n
        assert k_max == obj.k_max
        assert k == 3
        return sentinel

    monkeypatch.setattr(core_sg_module, "core_sg_mutual_reachability_distance", fake_core_sg_mutual_reachability_distance)

    result = obj.get_core_sg_mutual_reachability_distance(3)

    assert result is sentinel


def test_extract_hierarchy_for_smaller_k_updates_current_attributes(core_sg_module, fitted_obj, monkeypatch):
    obj, payload = fitted_obj
    mst_small = np.array(
        [
            [0, 1, 1.0],
            [1, 2, 1.4],
            [2, 3, 6.4],
            [3, 4, 1.0],
            [4, 5, 1.4],
        ],
        dtype=np.float64,
    )
    slt_small = mst_small + 0.5
    condensed = np.column_stack([np.arange(obj.n), np.arange(obj.n), np.ones(obj.n), np.full(obj.n, 2.0)]).astype(np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.linspace(0.5, 1.0, obj.n)
    persistence = np.array([0.55, 0.88], dtype=np.float64)

    monkeypatch.setattr(core_sg_module, "mst_from_core_sg", lambda **kwargs: mst_small)
    monkeypatch.setattr(core_sg_module, "label", lambda mst: slt_small)

    seen = {}

    def fake_tree_to_labels(instance, single_linkage_tree, min_spanning_tree):
        seen["kwargs"] = instance._get_tree_to_labels_kwargs()
        assert np.array_equal(single_linkage_tree, slt_small)
        assert np.array_equal(min_spanning_tree, mst_small)
        return labels, probabilities, persistence, condensed, single_linkage_tree, min_spanning_tree

    monkeypatch.setattr(core_sg_module, "tree_to_labels", fake_tree_to_labels)

    obj.extract_hierarchy_from_core_sg(3)

    assert seen["kwargs"] == {"cluster_selection_method": "leaf", "allow_single_cluster": True}
    assert np.array_equal(obj.labels_, labels)
    assert np.array_equal(obj.probabilities_, probabilities)
    assert np.array_equal(obj.cluster_persistence_, persistence)
    assert np.array_equal(obj._condensed_tree, condensed)
    assert np.array_equal(obj._single_linkage_tree, slt_small)
    assert np.array_equal(obj._min_spanning_tree, mst_small)

    assert obj.minimum_spanning_tree_.to_pandas().shape[0] == obj.n - 1
    assert obj.single_linkage_tree_.to_pandas().shape[0] == obj.n - 1
    assert obj.condensed_tree_.to_pandas().shape[0] >= obj.n


def test_extract_hierarchy_for_kmax_reuses_saved_fit_outputs(fitted_obj):
    obj, payload = fitted_obj

    obj.extract_hierarchy_from_core_sg(obj.k_max)

    assert np.array_equal(obj.labels_, obj.labels_k_max)
    assert np.array_equal(obj.probabilities_, obj.probabilities_k_max)
    assert np.array_equal(obj.cluster_persistence_, obj.cluster_persistence_k_max)
    assert np.array_equal(obj._condensed_tree, obj._condensed_tree_k_max)
    assert np.array_equal(obj._single_linkage_tree, obj._single_linkage_tree_k_max)
    assert np.array_equal(obj._min_spanning_tree, obj._min_spanning_tree_k_max)


def test_extract_mst_from_core_sg_uses_recomputed_path_for_smaller_k(core_sg_module, fitted_obj, monkeypatch):
    obj, _ = fitted_obj
    mst_small = np.array([[0, 1, 1.0], [1, 2, 1.2], [2, 3, 2.0], [3, 4, 1.1], [4, 5, 1.2]], dtype=np.float64)

    def fake_mst_from_core_sg(core_sg, metric_edges, core_k_list, n_nodes, k):
        assert core_sg is obj._core_sg
        assert metric_edges is obj._metric_edges
        assert core_k_list is obj._core_k_list
        assert n_nodes == obj.n
        assert k == 3
        return mst_small

    monkeypatch.setattr(core_sg_module, "mst_from_core_sg", fake_mst_from_core_sg)

    result = obj.extract_mst_from_core_sg(3)

    assert np.array_equal(result, mst_small)