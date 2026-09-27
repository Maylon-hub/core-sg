"""Public fast-path checks: do not force the rounded dense implementation."""
import numpy as np
import hdbscan
import pytest
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import pairwise_distances
import sys
import importlib


@pytest.fixture(autouse=True)
def real_core_package():
    # Unit doubles elsewhere leave already-imported CORE-SG adapters referring
    # to FakeHDBSCAN after monkeypatch restores sys.modules. Reload this test's
    # package boundary so these scientific oracles exercise real dependencies.
    for name in list(sys.modules):
        if name == 'core_sg' or name.startswith('core_sg.'):
            del sys.modules[name]
    package = importlib.import_module('core_sg')
    yield package


@pytest.mark.parametrize('k', [2, 4, 6])
def test_unrounded_public_tree_matches_reference(k):
    from core_sg import CoreSG
    X = np.random.default_rng(37).normal(size=(30, 3))
    model = CoreSG(no_noise=False).fit(X, k_max=6)
    model.extract_hierarchy_from_core_sg(k=k)
    ref = hdbscan.HDBSCAN(min_samples=k, min_cluster_size=k,
        match_reference_implementation=True, approx_min_span_tree=False,
        core_dist_n_jobs=1).fit(X)
    np.testing.assert_allclose(cophenet(model.single_linkage_tree_.to_numpy()),
                               cophenet(ref.single_linkage_tree_.to_numpy()), atol=1e-8)


def test_public_distance_matrix_and_mst_metric_weights_are_not_placeholders():
    from core_sg import CoreSG
    X = np.random.default_rng(21).normal(size=(20, 3))
    model = CoreSG(no_noise=False).fit(X, k_max=6)
    np.testing.assert_allclose(model.distance_matrix_, pairwise_distances(X), atol=1e-12)
    edges = model.metric_edges_
    expected = np.linalg.norm(X[edges[:, 0].astype(int)] - X[edges[:, 1].astype(int)], axis=1)
    np.testing.assert_allclose(edges[:, 2], expected, atol=1e-12)
