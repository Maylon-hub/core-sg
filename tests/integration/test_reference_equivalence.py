from __future__ import annotations

import numpy as np
import pytest

hdbscan = pytest.importorskip("hdbscan")
from sklearn.datasets import make_blobs

from core_sg import CoreSG
from tests.helpers import (
    normalize_undirected_edges,
    validate_reference_edges_in_core,
    validate_reference_weights_in_core,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset():
    X, _ = make_blobs(n_samples=60, n_features=3, centers=4, random_state=42)
    return X


def _reference_mst(D: np.ndarray, k: int) -> np.ndarray:
    ref = hdbscan.HDBSCAN(
        min_cluster_size=k,
        min_samples=k,
        metric="precomputed",
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        match_reference_implementation=True,
    ).fit(D)
    mst = np.asarray(ref._min_spanning_tree, dtype=np.float64)
    return mst[np.argsort(mst[:, 2], kind="mergesort")]


class TestReferenceEquivalence:
    @pytest.mark.parametrize("k", [6, 4, 2])
    def test_reference_mst_edges_are_contained_in_core_sg(self, dataset, k):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)

        mst_hdb = _reference_mst(core.distance_matrix_, k)
        summary = validate_reference_edges_in_core(core.support_graph_, mst_hdb)

        assert summary.ok, (
            f"Missing {summary.missing} reference edges out of {summary.compared}."
        )

    @pytest.mark.parametrize("k", [6, 4, 2])
    def test_reference_mrd_weights_are_present_in_reweighted_core_sg(self, dataset, k):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)

        mst_hdb = _reference_mst(core.distance_matrix_, k)
        weighted_core = core.get_core_sg_mutual_reachability_distance(k)
        summary = validate_reference_weights_in_core(weighted_core, mst_hdb)

        assert summary.ok, (
            f"Missing {summary.missing} weighted reference edges out of "
            f"{summary.compared}."
        )

    @pytest.mark.parametrize("k", [6, 4, 2])
    def test_reference_and_core_sg_mst_are_reported_without_failing(
        self, dataset, k, capsys
    ):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)

        mst_hdb = _reference_mst(core.distance_matrix_, k)
        mst_core = core.extract_mst_from_core_sg(k)

        same_weighted_edges = normalize_undirected_edges(
            mst_hdb
        ) == normalize_undirected_edges(mst_core)
        print(f"k={k} | exact weighted match={same_weighted_edges}")

        captured = capsys.readouterr()
        assert f"k={k}" in captured.out
        assert "exact weighted match=" in captured.out


class TestIntegrationSmoke:
    def test_public_package_import_exposes_core_sg_class(self):
        from core_sg import CoreSG as ExportedCoreSG

        assert ExportedCoreSG.__name__ == CoreSG.__name__ == "CoreSG"
        assert ExportedCoreSG.__module__ == CoreSG.__module__ == "core_sg.core_sg"

    def test_full_fit_extract_hierarchy_and_wrapped_accessors_with_real_hdbscan(
        self, dataset
    ):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)
        core.extract_hierarchy_from_core_sg(4)

        assert core.labels_ is not None
        assert core.probabilities_ is not None
        assert core.cluster_persistence_ is not None
        assert core.condensed_tree_.to_pandas().shape[0] >= core.n_samples_
        assert core.single_linkage_tree_.to_pandas().shape[0] == core.n_samples_ - 1
        assert core.minimum_spanning_tree_.to_pandas().shape[0] == core.n_samples_ - 1

    def test_score_sg_fit_extract_hierarchy_and_wrapped_accessors(self, dataset):
        connected_dataset, _ = make_blobs(
            n_samples=60,
            n_features=3,
            centers=1,
            cluster_std=1.0,
            random_state=42,
        )
        core = CoreSG(
            metric="euclidean",
            p=2,
            algorithm="score-sg",
            random_state=42,
            verbose=0,
            match_reference_implementation=True,
        )
        core._fit_for_tests(connected_dataset, 6)
        core.extract_hierarchy_from_core_sg(4)

        with pytest.raises(
            AttributeError,
            match="Attribute 'distance_matrix_' is available only when algorithm='core-sg'",
        ):
            _ = core.distance_matrix_
        assert core._tree_to_labels_data_.shape == connected_dataset.shape
        assert core.anti_hubs_ is not None
        assert core.anti_hubs_.ndim == 1
        assert core.anti_hubs_.shape[0] == int(
            np.floor(np.sqrt(connected_dataset.shape[0]))
        )
        assert core.labels_ is not None
        assert core.probabilities_ is not None
        assert core.cluster_persistence_ is not None
        assert core.condensed_tree_.to_pandas().shape[0] >= core.n_samples_
        assert core.single_linkage_tree_.to_pandas().shape[0] == core.n_samples_ - 1
        assert core.minimum_spanning_tree_.to_pandas().shape[0] == core.n_samples_ - 1

    def test_score_sg_fit_raises_explicitly_when_support_graph_is_disconnected(
        self, dataset
    ):
        core = CoreSG(
            metric="euclidean",
            p=2,
            algorithm="score-sg",
            random_state=42,
            verbose=0,
            match_reference_implementation=True,
        )

        with pytest.raises(ValueError, match="support graph is disconnected"):
            core._fit_for_tests(dataset, 6)
