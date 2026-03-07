# python -m tests.validate_mst_weight --n 5000 --d 10 --centers 10 --k 15 --match-ref
from __future__ import annotations

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs

import hdbscan


from core_sg.core_sg import build_core_sg_from_data, mst_from_core_sg

from core_sg.validate import validate_mst_from_core_sg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--centers", type=int, default=10)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--match-ref", action="store_true")
    args = ap.parse_args()

    n = args.n
    d = args.d
    k = args.k
    seed = args.seed

    print(f"Generating synthetic dataset: n={n}, d={d}, centers={args.centers}, seed={seed}")
    X, _ = make_blobs(n_samples=n, n_features=d, centers=args.centers, random_state=seed)

    # --- Build Core-SG (calculando pairwise dentro) ---
    t0 = time.time()
    core_sg, metric_edges, core_k, D = build_core_sg_from_data(
        X,
        k_max=k,
        metric="euclidean",
        pairwise_dtype=np.float64,
        match_reference_implementation=args.match_ref,
    )
    t1 = time.time()
    print(f"Core-SG build done in {t1 - t0:.2f}s")
    #D[3881][2386] = 2.43

    for k_iter in range(k,2,-2):

        # --- MST final via Core-SG ---
        t2 = time.time()
        mst_core = mst_from_core_sg(
            core_sg=core_sg,
            metric_edges=metric_edges,
            core_k=core_k,
            n_nodes=n,
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
            match_reference_implementation=args.match_ref,
        ).fit(D)
        mst_hdb = np.asarray(ref._min_spanning_tree, dtype=np.float64)
        mst_hdb = mst_hdb[np.argsort(mst_hdb[:, 2], kind="mergesort")]
        t5 = time.time()
        print(f"HDBSCAN reference done in {t5 - t4:.2f}s")

        #1095 Minimo Diferente - HDB :  [2.38600000e+03 3.88100000e+03 2.42877254e+00] || Core:  [2.57800000e+03 3.88100000e+03 2.42877254e+00]
        #print("Results")
        #print(D[3881][2386],core_k[3881],core_k[2386])
        #print(D[3881][2578],core_k[3881],core_k[2578])

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


if __name__ == "__main__":
    main()