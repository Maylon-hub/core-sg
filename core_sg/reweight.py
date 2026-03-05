from __future__ import annotations
import numpy as np


def reweight_core_sg_mutual_reachability(
    core_sg: np.ndarray,
    core_k: np.ndarray,
    metric_edges: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    """
    Atualiza pesos do core_sg:
      w(u,v) = max(core_k[u], core_k[v], dist(u,v))
    usando lookup vetorizado em metric_edges (bigger, smaller, dist).

    Retorna (E,3) float64.
    """
    assert np.all(core_sg[:,2] >= -1)
    e = np.ascontiguousarray(core_sg, dtype=np.float64)
    u = e[:, 0].astype(np.int64, copy=False)
    v = e[:, 1].astype(np.int64, copy=False)

    bigger = np.maximum(u, v)
    smaller = np.minimum(u, v)

    me = np.ascontiguousarray(metric_edges, dtype=np.float64)
    me_b = me[:, 0].astype(np.int64, copy=False)
    me_s = me[:, 1].astype(np.int64, copy=False)
    me_w = me[:, 2].astype(np.float64, copy=False)

    base = int(n_nodes)
    key_core = bigger * base + smaller
    key_me = me_b * base + me_s

    order = np.argsort(key_me, kind="mergesort")
    key_me_sorted = key_me[order]
    w_sorted = me_w[order]

    pos = np.searchsorted(key_me_sorted, key_core)
    ok = (pos < key_me_sorted.size) & (key_me_sorted[pos] == key_core)
    if not np.all(ok):
        raise KeyError("Arestas do core_sg faltando em metric_edges (faltou merge MST→metric_edges?).")

    dist_uv = w_sorted[pos]

    ck = np.asarray(core_k, dtype=np.float64)
    e[:, 2] = np.maximum(np.maximum(ck[u], ck[v]), dist_uv)
    assert np.all(e[:,2] != -1), "Placeholder values not overwritten"
    return e