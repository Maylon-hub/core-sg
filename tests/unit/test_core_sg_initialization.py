from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestCoreSGInitialization:
    def test_init_sets_defaults_and_empty_state(self, core_sg_module):
        obj = core_sg_module.CoreSG()

        assert obj.metric == "euclidean"
        assert obj.p == 2
        assert obj.debug is False
        assert obj.no_noise is True
        assert obj.noise_label_strategy == "mst_label_propagation"
        assert obj.algorithm == "core-sg"
        assert obj.random_state is None
        assert obj.approx_knn_kwargs is None
        assert obj.hdbscan_kwargs == {}

        assert obj.n is None
        assert obj.k_max is None
        assert obj._core_sg is None
        assert obj._metric_edges is None
        assert obj._core_k_list is None
        assert obj._D is None
        assert obj._tree_to_labels_data is None
        with pytest.raises(
            AttributeError,
            match="Attribute 'anti_hubs_' is available only when algorithm='score-sg'",
        ):
            _ = obj.anti_hubs_

        assert obj.labels_ is None
        assert obj.probabilities_ is None
        assert obj.cluster_persistence_ is None
        assert obj._condensed_tree is None
        assert obj._single_linkage_tree is None
        assert obj._min_spanning_tree is None

    def test_init_accepts_noise_configuration(self, core_sg_module):
        obj = core_sg_module.CoreSG(
            no_noise=False,
            noise_label_strategy="mst_label_propagation",
        )

        assert obj.no_noise is False
        assert obj.noise_label_strategy == "mst_label_propagation"

    def test_init_rejects_non_boolean_no_noise(self, core_sg_module):
        with pytest.raises(TypeError, match="no_noise must be a boolean"):
            core_sg_module.CoreSG(no_noise="yes")

    def test_init_rejects_non_string_noise_label_strategy(self, core_sg_module):
        with pytest.raises(TypeError, match="noise_label_strategy must be a string"):
            core_sg_module.CoreSG(noise_label_strategy=123)

    def test_init_rejects_unknown_noise_label_strategy(self, core_sg_module):
        with pytest.raises(ValueError, match="Unknown noise_label_strategy"):
            core_sg_module.CoreSG(noise_label_strategy="unknown_strategy")

    def test_init_accepts_algorithm_random_state_and_approx_kwargs(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(
            algorithm="score-sg",
            random_state=7,
            approx_knn_kwargs={"n_trees": 4},
        )

        assert obj.algorithm == "score-sg"
        assert obj.random_state == 7
        assert obj.approx_knn_kwargs == {"n_trees": 4}

    def test_init_rejects_invalid_algorithm(self, core_sg_module):
        with pytest.raises(ValueError, match="algorithm must be one of"):
            core_sg_module.CoreSG(algorithm="unknown")

    def test_init_rejects_non_dict_approx_knn_kwargs(self, core_sg_module):
        with pytest.raises(TypeError, match="approx_knn_kwargs must be a dictionary"):
            core_sg_module.CoreSG(approx_knn_kwargs=["bad"])

    def test_score_sg_blocks_access_to_core_sg_specific_D_attribute(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(algorithm="score-sg")

        with pytest.raises(
            AttributeError,
            match="Attribute '_D' is available only when algorithm='core-sg'",
        ):
            _ = obj._D

    def test_score_sg_allows_access_to_anti_hubs_attribute_before_fit(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(algorithm="score-sg")

        assert obj.anti_hubs_ is None

    def test_get_tree_to_labels_kwargs_filters_supported_non_none_values(
        self, core_sg_module
    ):
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

    def test_properties_raise_before_fit_or_extract(self, core_sg_module):
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


class TestBuildCoreSGInputValidation:
    def test_build_core_sg_from_data_rejects_single_point_input(self, core_sg_module):
        X = np.array([[0.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="shape > 1"):
            core_sg_module.build_core_sg_from_data(X, 1)

    def test_build_core_sg_from_data_rejects_non_positive_k_max(self, core_sg_module):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="1 <= k_max <= n-1"):
            core_sg_module.build_core_sg_from_data(X, 0)

    def test_build_core_sg_from_data_rejects_k_max_greater_than_or_equal_to_n(
        self, core_sg_module
    ):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="1 <= k_max <= n-1"):
            core_sg_module.build_core_sg_from_data(X, 3)

    def test_build_core_sg_from_data_rejects_k_max_smaller_than_two(
        self, core_sg_module
    ):
        X = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="must be >= 2"):
            core_sg_module.build_core_sg_from_data(X, 1)
