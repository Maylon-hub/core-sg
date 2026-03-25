from __future__ import annotations

import pytest


def test_init_sets_defaults_and_empty_state(core_sg_module):
    obj = core_sg_module.CoreSG()

    assert obj.metric == "euclidean"
    assert obj.p == 2
    assert obj.debug is False
    assert obj.hdbscan_kwargs == {}

    assert obj.n is None
    assert obj.k_max is None
    assert obj._core_sg is None
    assert obj._metric_edges is None
    assert obj._core_k_list is None
    assert obj._D is None

    assert obj.labels_ is None
    assert obj.probabilities_ is None
    assert obj.cluster_persistence_ is None
    assert obj._condensed_tree is None
    assert obj._single_linkage_tree is None
    assert obj._min_spanning_tree is None


def test_get_tree_to_labels_kwargs_filters_supported_non_none_values(core_sg_module):
    obj = core_sg_module.CoreSG(
        cluster_selection_method="leaf",
        allow_single_cluster=True,
        cluster_selection_epsilon=0.25,
        unsupported_flag="ignored",
        max_cluster_size=None,
    )

    filtered = obj._get_tree_to_labels_kwargs()

    assert filtered == {
        "cluster_selection_method": "leaf",
        "allow_single_cluster": True,
        "cluster_selection_epsilon": 0.25,
    }


def test_properties_raise_before_fit_or_extract(core_sg_module):
    obj = core_sg_module.CoreSG()

    with pytest.raises(AttributeError):
        _ = obj.minimum_spanning_tree_
    with pytest.raises(AttributeError):
        _ = obj.minimum_spanning_tree_k_max_
    with pytest.raises(AttributeError):
        _ = obj.single_linkage_tree_
    with pytest.raises(AttributeError):
        _ = obj.single_linkage_tree_k_max_
    with pytest.raises(AttributeError):
        _ = obj.condensed_tree_
    with pytest.raises(AttributeError):
        _ = obj.condensed_tree_k_max_
    with pytest.raises(AttributeError):
        _ = obj.get_fitted_hdbscan_objects()
