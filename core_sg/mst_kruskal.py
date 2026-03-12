# core_sg/mst_kruskal.py
from __future__ import annotations

import numpy as np

# Evita keyword "from". Mantém padrão claro e compatível.
MST_EDGE_DTYPE = np.dtype([("u", np.int64), ("v", np.int64), ("distance", np.float64)])


class UnionFind:
    """
    Union-Find with path compression + union by rank.
    """

    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        parent = self.parent
        # Path halving (geralmente mais rápido)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        parent = self.parent
        rank = self.rank

        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        # Union by rank
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True


def kruskal_mst(edges: np.ndarray, n_nodes: int) -> np.recarray:
    """
    Kruskal for edges (E,3): [u, v, w] (qualquer dtype numérico).

    Retorna:
      recarray with  dtype [('u', int64), ('v', int64), ('distance', float64)]
      and size n_nodes-1.

    """
    e = np.asarray(edges)
    if e.ndim != 2 or e.shape[1] < 3:
        raise ValueError("edges must have shape (E,3) with columns [u, v, w].")
    if n_nodes <= 1:
        raise ValueError("n_nodes must be >= 2.")

    # Colunas (sem forçar float64 em tudo; só o peso precisa ser float64)
    u_all = np.asarray(e[:, 0], dtype=np.int64)
    v_all = np.asarray(e[:, 1], dtype=np.int64)
    w_all = np.asarray(e[:, 2], dtype=np.float64)

    # Ordena por peso (Kruskal)
    order = np.lexsort((u_all,v_all, w_all))
    #order = np.argsort(w_all, kind="mergesort") #-> Gera Inconsistência com o HDBSCAN referencia
    u_all = u_all[order]
    v_all = v_all[order]
    w_all = w_all[order]

    uf = UnionFind(n_nodes)

    # Prealoca saída em vetores (muito mais rápido que recarray record-by-record)
    mst_u = np.empty(n_nodes - 1, dtype=np.int64)
    mst_v = np.empty(n_nodes - 1, dtype=np.int64)
    mst_w = np.empty(n_nodes - 1, dtype=np.float64)

    # Locals para performance
    union = uf.union
    m = 0

    # Loop principal
    for i in range(w_all.size):
        ui = int(u_all[i])
        vi = int(v_all[i])
        if union(ui, vi):
            mst_u[m] = ui
            mst_v[m] = vi
            mst_w[m] = w_all[i]
            m += 1
            if m == n_nodes - 1:
                break

    if m != n_nodes - 1:
        raise ValueError("Disconex Graph: MST extraction is not possible.")

    # Materializa structured array só no final (barato)
    out = np.empty(n_nodes - 1, dtype=MST_EDGE_DTYPE).view(np.recarray)
    out.u = mst_u
    out.v = mst_v
    out.distance = mst_w
    return out