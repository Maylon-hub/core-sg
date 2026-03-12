# python -m tests.validate_mst_weight --n 5000 --d 10 --centers 10 --k 50
from __future__ import annotations

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs

import hdbscan


from core_sg.core_sg import build_core_sg_from_data, mst_from_core_sg

from tests.validate import validate_mst_from_core_sg

def test_mst_weights():
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

    for k_iter in range(k,2,-2):
        
        # --- MST final via Core-SG ---
        t2 = time.time()
        mst_core = mst_from_core_sg(
            core_sg=core_sg,
            metric_edges=metric_edges,
            core_k_list=core_k_list,
            n_nodes=n,
            k=k_iter
        )
        t3 = time.time()
        print(f"Core-SG MST (Kruskal) done in {t3 - t2:.2f}s")
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


        # --- Comparação MST: arestas + pesos ---    
        obj = validate_mst_from_core_sg(
            mst_core,
            mst_hdb,
            n,
            k_iter
        )

        if not obj.ok:
            print(obj)
            raise ValueError(f"A MST para k = {k_iter} nao eh igual à extraída via HDBSCAN")


