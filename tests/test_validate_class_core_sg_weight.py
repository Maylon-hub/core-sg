# python -m tests.validate_class_core_sg_weight --n 5000 --d 10 --centers 10 --k 50 
from __future__ import annotations

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs

import hdbscan


from core_sg.core_sg import CoreSG
from tests.validate import validate_weights_in_core_sg


def test_class_core_sg_mst_mrd():
    n = 5000
    d = 10
    centers = 10
    k = 50
    seed = 42
    atol = 1e-12
    rtol = 1e-9

    print(f"Generating synthetic dataset: n={n}, d={d}, centers={centers}, seed={seed}")
    X, _ = make_blobs(
        n_samples=n,
        n_features=d,
        centers=centers,
        random_state=seed,
    )

    # --- Build Core-SG (calculando pairwise dentro) ---
    core_sg_ = CoreSG(metric='euclidean',p=2,debug=True,match_reference_implementation=True)
    core_sg_.fit(X,k,test_only=True)

    for k_iter in range(k,2,-2):

        # --- HDBSCAN referência (MST mutual reachability) ---
        t4 = time.time()
        ref = hdbscan.HDBSCAN(
            min_cluster_size=k_iter,
            min_samples=k_iter,
            metric="precomputed",
            algorithm="generic",
            approx_min_span_tree=False,
            gen_min_span_tree=True,
            match_reference_implementation=True,
        ).fit(core_sg_._D)
        mst_hdb = np.asarray(ref._min_spanning_tree, dtype=np.float64)
        mst_hdb = mst_hdb[np.argsort(mst_hdb[:, 2], kind="mergesort")]
        t5 = time.time()
        print(f"HDBSCAN reference done in {t5 - t4:.2f}s")


        weighted = core_sg_.get_core_sg_mutual_reachability_distance(k_iter)
        # --- Comparação MST: arestas + pesos ---  
        obj = validate_weights_in_core_sg(
            weighted,
            mst_hdb,
            core_sg_._D,
            core_sg_.get_core_distance(k_iter),
            n,
            k_iter
        )

        if not obj.ok:
            print(obj)
            raise ValueError(f"A MST para k = {k_iter} nao esta contida no Core-SG")


test_class_core_sg_mst_mrd()