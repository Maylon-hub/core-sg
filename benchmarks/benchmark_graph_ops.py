from __future__ import annotations

import argparse
import timeit

import numpy as np
from sklearn.datasets import make_blobs

from core_sg.core_sg import build_core_sg_from_data, mst_from_core_sg
from core_sg.mst_kruskal import kruskal_mst
from core_sg.reweight import reweight_core_sg_mutual_reachability


def benchmark_case(n_samples: int, k_max: int, n_features: int, centers: int) -> str:
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=42,
    )
    core_sg, metric_edges, core_k_list, _D, _ = build_core_sg_from_data(
        X, k_max=k_max, test_only=True
    )
    k = min(10, k_max)
    core_k = np.ascontiguousarray(core_k_list[:, k - 1], dtype=np.float64)
    weighted = reweight_core_sg_mutual_reachability(
        core_sg, core_k, metric_edges, n_nodes=n_samples
    )

    kruskal_time = min(
        timeit.repeat(lambda: kruskal_mst(weighted, n_nodes=n_samples), number=5, repeat=5)
    ) / 5
    reweight_first = timeit.timeit(
        lambda: reweight_core_sg_mutual_reachability(
            core_sg, core_k, metric_edges, n_nodes=n_samples
        ),
        number=1,
    )
    reweight_repeat = min(
        timeit.repeat(
            lambda: reweight_core_sg_mutual_reachability(
                core_sg, core_k, metric_edges, n_nodes=n_samples
            ),
            number=5,
            repeat=5,
        )
    ) / 5
    full_flow_time = min(
        timeit.repeat(
            lambda: mst_from_core_sg(core_sg, metric_edges, core_k_list, n_samples, k),
            number=1,
            repeat=5,
        )
    )
    return (
        f"n={n_samples} k_max={k_max} "
        f"kruskal={kruskal_time:.6f}s "
        f"reweight_first={reweight_first:.6f}s "
        f"reweight_repeat={reweight_repeat:.6f}s "
        f"flow={full_flow_time:.6f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["80:8", "300:15", "800:30"],
        help="Benchmark cases in the form n_samples:k_max.",
    )
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--centers", type=int, default=6)
    args = parser.parse_args()

    for case in args.cases:
        n_samples_str, k_max_str = case.split(":", maxsplit=1)
        print(
            benchmark_case(
                n_samples=int(n_samples_str),
                k_max=int(k_max_str),
                n_features=args.n_features,
                centers=args.centers,
            )
        )


if __name__ == "__main__":
    main()
