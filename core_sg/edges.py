from __future__ import annotations
import numpy as np

def sort_core_sg(core_sg: np.ndarray) -> np.ndarray:
    """
    Normaliza o Core-SG para o formato [menor_idx, maior_idx, distancia]
    e ordena por (distancia, maior_idx, menor_idx).
    """
    if core_sg.ndim != 2 or core_sg.shape[1] < 3:
        raise ValueError("mst deve ter shape (n_edges, 3)")

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

def build_knng_vectors(
    idxs_arr: np.ndarray,
    distance_arr: np.ndarray,
    knng_size: int,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
      metric_edges: (E,3) [bigger, smaller, dist]
      knng_to_insert: (E,3) [i, neighbor, dist]
    onde E = knng_size*k_max.
    """
    if idxs_arr.shape != (knng_size, k_max):
        raise ValueError(f"idxs_arr.shape deve ser {(knng_size, k_max)}, recebeu {idxs_arr.shape}")
    if distance_arr.shape != (knng_size, k_max):
        raise ValueError(f"distance_arr.shape deve ser {(knng_size, k_max)}, recebeu {distance_arr.shape}")

    idxs_arr = np.ascontiguousarray(idxs_arr, dtype=np.int64)
    distance_arr = np.ascontiguousarray(distance_arr, dtype=np.float64)

    idx_a = np.repeat(np.arange(knng_size, dtype=np.int64), k_max)
    neigh = idxs_arr.reshape(-1)
    dist = distance_arr.reshape(-1)

    knng_to_insert = np.empty((knng_size * k_max, 3), dtype=np.float64)
    knng_to_insert[:, 0] = idx_a
    knng_to_insert[:, 1] = neigh
    knng_to_insert[:, 2] = dist

    bigger = np.maximum(idx_a, neigh)
    smaller = np.minimum(idx_a, neigh)

    metric_edges = np.empty((knng_size * k_max, 3), dtype=np.float64)
    metric_edges[:, 0] = bigger
    metric_edges[:, 1] = smaller
    metric_edges[:, 2] = dist

    return metric_edges, knng_to_insert


def add_mst_edges_to_metric_edges(metric_edges: np.ndarray, mst: np.ndarray, *, n_nodes: int | None = None) -> np.ndarray:
    """
    Adiciona arestas da MST em metric_edges, sem repetição (pelo par bigger/smaller).
    metric_edges: (E,3) [bigger, smaller, dist]
    mst:          (M,3) [u, v, w] (qualquer ordem) OU [bigger, smaller, w]
    """
    me = np.ascontiguousarray(metric_edges, dtype=np.float64)
    mst = np.ascontiguousarray(mst, dtype=np.float64)

    if me.ndim != 2 or me.shape[1] < 3:
        raise ValueError("metric_edges deve ser (E,3).")
    if mst.ndim != 2 or mst.shape[1] < 3:
        raise ValueError("mst deve ser (M,3).")

    me_b = me[:, 0].astype(np.int64, copy=False)
    me_s = me[:, 1].astype(np.int64, copy=False)

    u = mst[:, 0].astype(np.int64, copy=False)
    v = mst[:, 1].astype(np.int64, copy=False)
    w = mst[:, 2].astype(np.float64, copy=False)

    mst_b = np.maximum(u, v)
    mst_s = np.minimum(u, v)

    if n_nodes is None:
        max_idx = int(
            max(
                me_b.max(initial=0),
                me_s.max(initial=0),
                mst_b.max(initial=0),
                mst_s.max(initial=0),
            )
        )
        n_nodes = max_idx + 1

    base = int(n_nodes)
    me_key = me_b * base + me_s
    me_key_sorted = np.sort(me_key, kind="mergesort")

    mst_key = mst_b * base + mst_s
    pos = np.searchsorted(me_key_sorted, mst_key)
    exists = (pos < me_key_sorted.size) & (me_key_sorted[pos] == mst_key)
    add_mask = ~exists

    if not np.any(add_mask):
        return me

    to_add = np.empty((int(add_mask.sum()), 3), dtype=np.float64)
    to_add[:, 0] = mst_b[add_mask]
    to_add[:, 1] = mst_s[add_mask]
    to_add[:, 2] = w[add_mask]

    return np.vstack([me, to_add])