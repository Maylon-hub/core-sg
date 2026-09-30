"""Qualify an installed CORE-SG artifact; run outside the source checkout."""
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import make_blobs


def main():
    modules = {name: importlib.import_module(name) for name in
               ("core_sg", "core_sg._mst_kruskal", "core_sg._reweight")}
    prefix = Path(sys.prefix).resolve()
    for name, module in modules.items():
        assert Path(module.__file__).resolve().is_relative_to(prefix), (name, module.__file__)
    for name in ("core_sg._mst_kruskal", "core_sg._reweight"):
        assert Path(modules[name].__file__).suffix in (".pyd", ".so")
    from hdbscan.hdbscan_ import _tree_to_labels
    assert metadata.version("hdbscan") == "0.8.44"
    assert {"min_cluster_size", "match_reference_implementation", "cluster_selection_method"} <= set(inspect.signature(_tree_to_labels).parameters)
    X, _ = make_blobs(n_samples=90, n_features=3, centers=3, random_state=42)
    for metric in ("euclidean", "manhattan"):
        model = modules["core_sg"].CoreSG(metric=metric, no_noise=False).fit(X, k_max=8)
        model.extract_hierarchy_from_core_sg(k=8)
        expected = model.single_linkage_tree_.to_numpy().copy()
        for k in (4, 6, 8):
            model.extract_hierarchy_from_core_sg(k=k)
            assert model.single_linkage_tree_.to_numpy().shape == (89, 4)
            assert model.labels_.shape == (90,)
        np.testing.assert_allclose(model.single_linkage_tree_.to_numpy(), expected)
    print(json.dumps({"python": sys.version, "prefix": str(prefix),
                      "versions": {n: metadata.version(n) for n in ("core-sg-mustache", "hdbscan", "numpy", "scipy", "scikit-learn")},
                      "modules": {n: m.__file__ for n, m in modules.items()}, "metrics": ["euclidean", "manhattan"], "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
