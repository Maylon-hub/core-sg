from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import hdbscan

from .core_sg import build_core_sg_from_data, mst_and_single_linkage_from_core_sg


@dataclass(frozen=True)
class CoreSGValidationReport:
    n: int
    k_max: int
    ok: bool
    same_edge_set: bool
    missing_in_core: int
    extra_in_core: int
    max_abs_weight_diff: float | None
    n_weight_mismatches: int | None


def _mst_keys_and_weights(mst_arr: np.ndarray, base: int) -> tuple[np.ndarray, np.ndarray]:
    u = mst_arr[:, 0].astype(np.int64, copy=False)
    v = mst_arr[:, 1].astype(np.int64, copy=False)
    w = mst_arr[:, 2].astype(np.float64, copy=False)
    b = np.maximum(u, v)
    s = np.minimum(u, v)
    key = b * base + s
    return key, w


def validate_core_sg(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    atol: float = 1e-12,
    rtol: float = 1e-9,
    **hdbscan_kwargs,
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    X = np.asarray(X)
    n = X.shape[0]
    base = n  # chave única safe: bigger*n + smaller

    # Core-SG + D (precomputed) gerada internamente
    core_sg, metric_edges, core_k, D = build_core_sg_from_data(
        X,
        k_max,
        metric=metric,
        p=p,
        include_mst_edges=True,
        **hdbscan_kwargs,
    )

    # HDBSCAN referência (precomputed) usando a MESMA matriz D
    D = np.asarray(D, dtype=np.float64)
    ref = hdbscan.HDBSCAN(
        min_cluster_size=k_max,
        min_samples=k_max,
        metric="precomputed",
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        **hdbscan_kwargs,
    ).fit(D)
    mst_ref = np.asarray(ref._min_spanning_tree, dtype=np.float64)

    # MST final via Core-SG
    mst_core, _slt_core = mst_and_single_linkage_from_core_sg(
        core_sg, metric_edges, core_k, n_nodes=n
    )

    ref_key, ref_w = _mst_keys_and_weights(mst_ref, base)
    core_key, core_w = _mst_keys_and_weights(mst_core, base)

    ref_set = set(ref_key.tolist())
    core_set = set(core_key.tolist())

    missing = len(ref_set - core_set)
    extra = len(core_set - ref_set)
    same_edge_set = (missing == 0 and extra == 0)

    max_abs_diff = None
    n_mismatch = None

    if same_edge_set:
        ref_map = dict(zip(ref_key.tolist(), ref_w.tolist()))
        core_map = dict(zip(core_key.tolist(), core_w.tolist()))

        diffs = []
        mism = 0
        for kk in ref_map.keys():
            a = float(ref_map[kk])
            b = float(core_map[kk])
            diffs.append(abs(a - b))
            if not np.isclose(a, b, atol=atol, rtol=rtol):
                mism += 1

        max_abs_diff = float(np.max(diffs)) if diffs else 0.0
        n_mismatch = int(mism)

    ok = same_edge_set and (n_mismatch == 0)
    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k_max),
        ok=bool(ok),
        same_edge_set=bool(same_edge_set),
        missing_in_core=int(missing),
        extra_in_core=int(extra),
        max_abs_weight_diff=max_abs_diff,
        n_weight_mismatches=n_mismatch,
    )