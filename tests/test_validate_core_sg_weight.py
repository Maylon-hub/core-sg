# python -m tests.validate_core_sg_weight --n 5000 --d 10 --centers 10 --k 50 
from __future__ import annotations

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs

import hdbscan


from core_sg.core_sg import build_core_sg_from_data

from tests.validate import validate_weights_in_core_sg
from core_sg.reweight import reweight_core_sg_mutual_reachability


def test_core_sg_mrd_weights():
    n = 5000
    d = 10
    centers = 10
    k = 30
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
    t0 = time.time()
    core_sg, metric_edges, core_k_list, D,_ = build_core_sg_from_data(
        X,
        k_max=k,
        metric="euclidean",
        pairwise_dtype=np.float64,
        test_only=True
    )
    t1 = time.time()
    print(f"Core-SG build done in {t1 - t0:.2f}s")
    #D[3881][2386] = 2.43

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
        ).fit(D)
        mst_hdb = np.asarray(ref._min_spanning_tree, dtype=np.float64)
        mst_hdb = mst_hdb[np.argsort(mst_hdb[:, 2], kind="mergesort")]
        t5 = time.time()
        print(f"HDBSCAN reference done in {t5 - t4:.2f}s")

        core_k = core_k_list[:, k_iter - 1]
        core_k = np.ascontiguousarray(core_k, dtype=np.float64)

        weighted = reweight_core_sg_mutual_reachability(
            core_sg=core_sg,
            core_k=core_k,
            metric_edges=metric_edges,
            n_nodes=n,
        )
        # --- Comparação MST: arestas + pesos ---  
        obj = validate_weights_in_core_sg(
            weighted,
            mst_hdb,
            D,
            core_k,
            n,
            k_iter
        )

        if not obj.ok:
            print(obj)
            raise ValueError(f"A MST para k = {k_iter} nao esta contida no Core-SG")


