# python -m tests.validate_class --n 5000 --d 10 --centers 10 --k 50 
from __future__ import annotations

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs


from core_sg.core_sg import CoreSG
from tests.validate import validate_core_sg_atributtes


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
    core_sg_ = CoreSG(metric='euclidean',p=2,debug=True,match_reference_implementation=True)
    core_sg_.fit(X,k,test_only=True)

    for k_iter in range(k,2,-2):
        try:
            core_sg_.extract_hierarchy_from_core_sg(k_iter)

            attributes_val = {'mst': core_sg_.minimum_spanning_tree_,
                              'mst_k': core_sg_.minimum_spanning_tree_k_max_,
                               'slt': core_sg_.single_linkage_tree_,
                                'slt_k': core_sg_.single_linkage_tree_k_max_,
                                'condensed': core_sg_.condensed_tree_,
                                'condensed_k': core_sg_.condensed_tree_k_max_,
            }
            
            for key,att in attributes_val.items():
                obj = validate_core_sg_atributtes(att,n,k_iter,key in ('condensed','condensed_k'))

                if not obj.ok:
                    raise RuntimeError(f"Erro na transformacao dos atributos em pandas. {obj}")
        except Exception as e:
            raise RuntimeError(f"Erro no uso da classe CoreSG {e}")


if __name__ == "__main__":
    main()