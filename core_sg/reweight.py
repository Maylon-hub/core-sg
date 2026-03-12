from __future__ import annotations
import numpy as np

def sort_core_sg(core_sg: np.ndarray) -> np.ndarray:
    """
    CorSG to format [menor_idx, maior_idx, distancia]
    SOrting By (distancia, maior_idx, menor_idx).
    """
    if core_sg.ndim != 2 or core_sg.shape[1] < 3:
        raise ValueError("mst must have shape (n_edges, 3)")

    u = core_sg[:, 0].astype(np.int64, copy=False)
    v = core_sg[:, 1].astype(np.int64, copy=False)
    w = core_sg[:, 2].astype(np.float64, copy=False)

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    core_sg_tmp = np.empty((core_sg.shape[0], 3), dtype=np.float64)
    core_sg_tmp[:, 0] = u_min
    core_sg_tmp[:, 1] = v_max
    core_sg_tmp[:, 2] = w

    order = np.lexsort((core_sg_tmp[:, 0], core_sg_tmp[:, 1], core_sg_tmp[:, 2]))
    core_sg_tmp = core_sg_tmp[order]

    return core_sg_tmp


def reweight_core_sg_mutual_reachability(
    core_sg: np.ndarray,
    core_k: np.ndarray,
    metric_edges: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    """
    Update Core-sg weights:
      w(u,v) = max(core_k[u], core_k[v], dist(u,v))
    Using vectorized lookup in metric_edges (bigger, smaller, dist).

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
        raise KeyError("CoreSG edges are missing in metric_edges.")

    dist_uv = w_sorted[pos]

    ck = np.asarray(core_k, dtype=np.float64)
    e[:, 2] = np.maximum(np.maximum(ck[u], ck[v]), dist_uv)
    assert np.all(e[:,2] != -1), "Placeholder values not overwritten"
    e = sort_core_sg(e)
    return e