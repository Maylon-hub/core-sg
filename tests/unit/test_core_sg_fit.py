from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import assert_dataframe_schema, make_fit_payload

pytestmark = pytest.mark.unit


@pytest.fixture()
def patched_fit_dependencies(core_sg_module, monkeypatch):
    payload = make_fit_payload()

    def fake_build_core_sg_from_data(X, k_max, metric, p, test_only=False):
        assert X.shape[0] == payload[3].shape[0]
        assert k_max == 4
        assert metric == "euclidean"
        assert p == 2
        assert test_only is True
        return payload

    monkeypatch.setattr(
        core_sg_module, "build_core_sg_from_data", fake_build_core_sg_from_data
    )
    return payload


class TestCoreSGFit:
    def test_fit_returns_self_and_stores_artifacts(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG(metric="euclidean", p=2, debug=False)

        returned = obj.fit(sample_X, 4, test_only=True)
        core_sg, metric_edges, core_k_list, D, hdb_obj = patched_fit_dependencies

        assert returned is obj
        assert obj.n == sample_X.shape[0]
        assert obj.k_max == 4
        assert obj._raw_data is sample_X
        assert np.array_equal(obj._core_sg, core_sg)
        assert np.array_equal(obj._metric_edges, metric_edges)
        assert np.array_equal(obj._core_k_list, core_k_list)
        assert np.array_equal(obj._D, D)

        assert np.array_equal(obj.labels_k_max, hdb_obj.labels_)
        assert np.array_equal(obj.probabilities_k_max, hdb_obj.probabilities_)
        assert np.array_equal(
            obj.cluster_persistence_k_max, hdb_obj.cluster_persistence_
        )
        assert np.array_equal(
            obj._single_linkage_tree_k_max, hdb_obj._single_linkage_tree
        )
        assert np.array_equal(obj._min_spanning_tree_k_max, hdb_obj._min_spanning_tree)

    def test_extract_mst_from_core_sg_returns_cached_mst_for_k_max(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj.fit(sample_X, 4, test_only=True)

        result = obj.extract_mst_from_core_sg(4)

        assert np.array_equal(result, obj._min_spanning_tree_k_max)

    def test_extract_mst_from_core_sg_to_dataframe_has_expected_schema(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj.fit(sample_X, 4, test_only=True)

        df = obj.extract_mst_from_core_sg(4, toDF=True)

        assert_dataframe_schema(df, ["to", "from", "weight"])
        assert str(df["to"].dtype) == "int64"
        assert str(df["from"].dtype) == "int64"
        assert str(df["weight"].dtype) == "float64"

    def test_get_fitted_hdbscan_objects_raw_returns_saved_arrays(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj.fit(sample_X, 4, test_only=True)

        fitted = obj.get_fitted_hdbscan_objects(wrapped=False)

        assert set(fitted) == {
            "labels_",
            "probabilities_",
            "cluster_persistence_",
            "condensed_tree_",
            "single_linkage_tree_",
            "minimum_spanning_tree_",
        }
        assert np.array_equal(fitted["labels_"], obj.labels_k_max)
        assert np.array_equal(
            fitted["minimum_spanning_tree_"], obj._min_spanning_tree_k_max
        )

    def test_get_fitted_hdbscan_objects_wrapped_returns_hdbscan_like_wrappers(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj.fit(sample_X, 4, test_only=True)

        fitted = obj.get_fitted_hdbscan_objects(wrapped=True)

        assert fitted["condensed_tree_"].to_pandas().shape[0] >= obj.n
        assert fitted["single_linkage_tree_"].to_pandas().shape[0] == obj.n - 1
        assert fitted["minimum_spanning_tree_"].to_pandas().shape[0] == obj.n - 1

    def test_minimum_spanning_tree_fit_wrapper_warns_without_raw_data(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj.fit(sample_X, 4, test_only=True)
        obj._raw_data = None

        with pytest.warns(UserWarning, match="No raw data is available"):
            wrapped = obj.minimum_spanning_tree_k_max_

        assert wrapped is None
