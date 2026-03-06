from __future__ import annotations

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs

import hdbscan


from core_sg.core_sg import build_core_sg_from_data, mst_and_single_linkage_from_core_sg,reweight_core_sg_mutual_reachability
from core_sg.validate import _mst_keys_and_weights  # se for “privado”, você pode copiar a lógica aqui


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
        include_mst_edges=True,
        pairwise_dtype=np.float64,
        match_reference_implementation=args.match_ref,
    )
    t1 = time.time()
    print(f"Core-SG build done in {t1 - t0:.2f}s")
    print(f"Distance matrix shape: {D.shape}, dtype={D.dtype}")

    # --- MST final via Core-SG ---
    t2 = time.time()
    mst_core, _slt = mst_and_single_linkage_from_core_sg(
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
        min_cluster_size=k,
        min_samples=k,
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

    # --- Comparação MST: arestas + pesos ---
    base = n

    d_hdb = {}
    for hdb in core_sg:
        max_val = max(hdb[:2])
        min_val = min(hdb[:2])
        if max_val not in d_hdb:
            d_hdb[max_val] = {}
        d_hdb[max_val][min_val] = [hdb[2]]
    a = 0
    for core in mst_hdb:
        max_val = max(core[:2])
        min_val = min(core[:2])
        try:
            d_hdb[max_val][min_val].append(core[2])
        except:
            a += 1


    ref_key, ref_w = _mst_keys_and_weights(mst_hdb, base)
    core_key, core_w = _mst_keys_and_weights(mst_core, base)

    ref_set = set(ref_key.tolist())
    core_set = set(core_key.tolist())

    missing = ref_set - core_set
    extra = core_set - ref_set

    if missing or extra:
        print("FAIL: Edge set differs")
        print(f"{len(ref_set)} || {len(core_set)}")
        print(f"  missing_in_core={len(missing)}")
        print(f"  extra_in_core={len(extra)}")
        raise SystemExit(1)

    # alinha pesos
    ref_map = dict(zip(ref_key.tolist(), ref_w.tolist()))
    core_map = dict(zip(core_key.tolist(), core_w.tolist()))

    mism = 0
    max_abs = 0.0
    for kk, wref in ref_map.items():
        wcore = core_map[kk]
        diff = abs(wref - wcore)
        if diff > max_abs:
            max_abs = diff
        if not np.isclose(wref, wcore, atol=args.atol, rtol=args.rtol):
            mism += 1

    if mism == 0:
        print("PASS: Core-SG MST matches HDBSCAN MST (edges + weights).")
        print(f"max_abs_weight_diff={max_abs:.3e}")
        raise SystemExit(0)

    print("FAIL: same edges but weight mismatches")
    print(f"  n_weight_mismatches={mism}")
    print(f"  max_abs_weight_diff={max_abs:.3e}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()