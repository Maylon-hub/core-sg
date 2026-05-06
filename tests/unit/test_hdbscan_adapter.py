from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def hdbscan_adapter_module(fake_hdbscan_modules):
    return importlib.import_module("core_sg.hdbscan_adapter")


class TestHDBSCANAdapter:
    def test_reference_mst_original_distance_uses_expected_hdbscan_parameters(
        self, hdbscan_adapter_module
    ):
        D = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )

        clusterer, mst = hdbscan_adapter_module.reference_mst_original_distance(
            D, k_max=2
        )

        assert clusterer.kwargs == {
            "min_cluster_size": 2,
            "min_samples": 2,
            "metric": "precomputed",
            "algorithm": "generic",
            "approx_min_span_tree": False,
            "gen_min_span_tree": True,
            "match_reference_implementation": True,
        }
        assert mst.dtype == np.float64
        assert mst.shape == (D.shape[0] - 1, 3)

    def test_mst_to_single_linkage_tree_delegates_to_hdbscan_label(
        self, hdbscan_adapter_module
    ):
        mst = np.array([[0, 1, 1.0], [1, 2, 2.0]], dtype=np.float64)

        result = hdbscan_adapter_module.mst_to_single_linkage_tree(mst)

        assert np.array_equal(result, mst)
        assert result.dtype == np.float64

    def test_tree_to_labels_forwards_tree_kwargs_and_appends_mst(
        self, hdbscan_adapter_module
    ):
        data = np.zeros((4, 2), dtype=np.float64)
        single_linkage_tree = np.array(
            [[0, 1, 1.0], [1, 2, 2.0], [2, 3, 3.0]], dtype=np.float64
        )
        min_spanning_tree = single_linkage_tree.copy()

        result = hdbscan_adapter_module.tree_to_labels(
            data,
            single_linkage_tree,
            tree_kwargs={
                "cluster_selection_method": "leaf",
                "allow_single_cluster": True,
            },
            min_spanning_tree=min_spanning_tree,
        )

        labels, probabilities, persistence, condensed, slt, mst = result
        assert np.array_equal(labels, np.arange(data.shape[0]) % 2)
        assert probabilities.shape == (data.shape[0],)
        assert persistence.shape == (2,)
        assert condensed.shape[0] == data.shape[0]
        assert np.array_equal(slt, single_linkage_tree)
        assert np.array_equal(mst, min_spanning_tree)

    def test_wrapper_functions_return_hdbscan_style_wrappers(
        self, hdbscan_adapter_module
    ):
        raw_tree = np.array([[0, 1, 1.0]], dtype=np.float64)
        labels = np.array([0, 0], dtype=np.int64)
        raw_data = np.array([[0.0], [1.0]], dtype=np.float64)

        condensed = hdbscan_adapter_module.wrap_condensed_tree(raw_tree, labels)
        single_linkage = hdbscan_adapter_module.wrap_single_linkage_tree(raw_tree)
        mst = hdbscan_adapter_module.wrap_minimum_spanning_tree(raw_tree, raw_data)

        assert np.array_equal(condensed.raw, raw_tree)
        assert np.array_equal(condensed.aux, labels)
        assert np.array_equal(single_linkage.raw, raw_tree)
        assert single_linkage.aux is None
        assert np.array_equal(mst.raw, raw_tree)
        assert np.array_equal(mst.aux, raw_data)


def test_core_sg_module_does_not_import_hdbscan_private_apis():
    source = Path("core_sg/core_sg.py").read_text()

    assert "from hdbscan" not in source
    assert "import hdbscan" not in source
    assert "hdbscan._hdbscan_linkage" not in source
    assert "hdbscan.hdbscan_" not in source
    assert "hdbscan.plots" not in source
