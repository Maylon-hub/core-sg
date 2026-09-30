"""Real dependency contract; no fake native-tree API in this check."""
from importlib.metadata import version
from inspect import signature


def test_qualified_private_hdbscan_signature():
    from hdbscan.hdbscan_ import _tree_to_labels
    assert version("hdbscan") == "0.8.44"
    assert {"min_cluster_size", "match_reference_implementation", "cluster_selection_method"} <= set(signature(_tree_to_labels).parameters)
