import numpy as np
from __future__ import annotations
from typing import Any

def knn_from_precomputed(D: np.ndarray, k: int, include_self: bool = False):
    """
    D: matriz NxN (distâncias). Assume D[i,i]=0 e valores menores = mais próximos.
    k: número de vizinhos desejado (se include_self=False, retorna k vizinhos excluindo i).
    include_self: se True, permite que i apareça como vizinho (normalmente False).
    
    Retorna:
      idx: (N, k) índices dos vizinhos
      dist: (N, k) distâncias correspondentes
    """
    D = np.asarray(D)
    n = D.shape[0]
    assert D.shape == (n, n), "D deve ser NxN"

    # Trabalhar numa cópia para não alterar D
    Dwork = D.copy()

    if not include_self:
        # impede escolher o próprio ponto como vizinho
        np.fill_diagonal(Dwork, np.inf)

    if k <= 0 or k > n - (0 if include_self else 1):
        raise ValueError("k inválido para o tamanho de D")

    # pega k menores por linha (não ordenados)
    idx_part = np.argpartition(Dwork, kth=k-1, axis=1)[:, :k]

    # agora ordena esses k por distância
    dist_part = np.take_along_axis(Dwork, idx_part, axis=1)
    order = np.argsort(dist_part, axis=1)

    idx = np.take_along_axis(idx_part, order, axis=1)
    dist = np.take_along_axis(dist_part, order, axis=1)

    return idx, dist


def build_knng_vectors(
    idxs_arr: np.ndarray,
    distance_arr: np.ndarray,
    knng_size: int,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói dois arrays vetorizados:

    1) knng_to_insert: (knng_size*k_max, 3)
       [idx_a, vizinho, distancia]

    2) metric_edges: (knng_size*k_max, 3)
       [bigger_idx, smaller_idx, distancia]

       onde:
           bigger_idx  = max(idx_a, vizinho)
           smaller_idx = min(idx_a, vizinho)

    Não há deduplicação de pares.

    Parameters
    ----------
    idxs_arr : ndarray (knng_size, k_max)
        Índices dos k vizinhos de cada ponto.
    distance_arr : ndarray (knng_size, k_max)
        Distâncias correspondentes.
    knng_size : int
    k_max : int

    Returns
    -------
    metric_edges : ndarray (knng_size*k_max, 3)
    knng_to_insert : ndarray (knng_size*k_max, 3)
    """

    # -------- validações --------

    if idxs_arr.shape != (knng_size, k_max):
        raise ValueError(
            f"idxs_arr.shape deve ser {(knng_size, k_max)}, recebeu {idxs_arr.shape}"
        )

    if distance_arr.shape != (knng_size, k_max):
        raise ValueError(
            f"distance_arr.shape deve ser {(knng_size, k_max)}, recebeu {distance_arr.shape}"
        )

    # -------- garantir tipos e contiguidade --------

    idxs_arr = np.ascontiguousarray(idxs_arr, dtype=np.int64)
    distance_arr = np.ascontiguousarray(distance_arr, dtype=np.float64)

    # -------- construir vetores flat --------

    idx_a = np.repeat(np.arange(knng_size, dtype=np.int64), k_max)
    vizinhos = idxs_arr.reshape(-1)
    dist = distance_arr.reshape(-1)

    # -------- knng_to_insert --------

    knng_to_insert = np.empty((knng_size * k_max, 3), dtype=np.float64)

    knng_to_insert[:, 0] = idx_a
    knng_to_insert[:, 1] = vizinhos
    knng_to_insert[:, 2] = dist

    # -------- metric_edges (triangular) --------

    bigger = np.maximum(idx_a, vizinhos)
    smaller = np.minimum(idx_a, vizinhos)

    metric_edges = np.empty((knng_size * k_max, 3), dtype=np.float64)

    metric_edges[:, 0] = bigger
    metric_edges[:, 1] = smaller
    metric_edges[:, 2] = dist

    return metric_edges, knng_to_insert



def mst_with_precomputed_distances(
    min_spanning_tree: Any,
    distance_matrix: np.ndarray,
    *,
    copy: bool = True,
) -> np.ndarray:
    """
    Substitui o "peso" (coluna de distância) do _min_spanning_tree do HDBSCAN
    pelas distâncias corretas vindas da matriz precomputed (distance_matrix).

    Observação importante:
    - O HDBSCAN armazena na MST distâncias de *mutual reachability* (ou variação),
      não necessariamente a distância original. Aqui trocamos o peso pelo valor
      distance_matrix[u, v] para cada aresta (u, v) da MST.

    Parâmetros
    ----------
    min_spanning_tree:
        Saída do HDBSCAN (ex.: clusterer._min_spanning_tree). Pode vir como:
        - ndarray (n-1, 3): [u, v, weight]
        - recarray/structured array com campos, tipicamente contendo u, v, weight
          (varia por versão).
    distance_matrix:
        Matriz NxN de distâncias (precomputed), onde distance_matrix[i, j] é a
        distância original entre i e j.
    copy:
        Se True, retorna uma cópia; se False e o tipo permitir, pode editar in-place.

    Retorna
    -------
    mst_out : ndarray (n-1, 3) float64
        Colunas: [u, v, distance_matrix[u, v]]
    """
    D = np.asarray(distance_matrix)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix deve ser uma matriz NxN.")
    n = D.shape[0]

    mst = min_spanning_tree

    # 1) Extrair u, v, w de forma robusta (dependendo do formato da MST)
    # Caso A: ndarray simples (n-1, 3)
    if isinstance(mst, np.ndarray) and mst.dtype.fields is None:
        mst_arr = np.asarray(mst)
        if mst_arr.ndim != 2 or mst_arr.shape[1] < 2:
            raise ValueError("min_spanning_tree ndarray deve ter ao menos 2 colunas (u, v).")
        u = mst_arr[:, 0].astype(np.int64, copy=False)
        v = mst_arr[:, 1].astype(np.int64, copy=False)

        # cria output padronizado (n-1, 3)
        if copy:
            out = np.empty((mst_arr.shape[0], 3), dtype=np.float64)
        else:
            # se quiser in-place e tiver 3 colunas, reaproveita
            if mst_arr.shape[1] >= 3 and np.issubdtype(mst_arr.dtype, np.floating):
                out = mst_arr
            else:
                out = np.empty((mst_arr.shape[0], 3), dtype=np.float64)

        out[:, 0] = u
        out[:, 1] = v

    # Caso B: structured / recarray (campos)
    else:
        mst_arr = np.asarray(mst)
        if mst_arr.dtype.fields is None:
            raise ValueError("Formato de min_spanning_tree não suportado.")

        names = list(mst_arr.dtype.names)

        # Heurística: detectar campos para u, v
        # (varia entre versões; tentamos opções comuns)
        def pick_field(candidates: list[str]) -> str:
            for c in candidates:
                if c in names:
                    return c
            return ""

        u_field = pick_field(["from", "u", "parent", "source", "i"])
        v_field = pick_field(["to", "v", "child", "target", "j"])

        # Se não achou, tenta fallback pelos 2 primeiros campos
        if not u_field or not v_field:
            u_field, v_field = names[0], names[1]

        u = mst_arr[u_field].astype(np.int64, copy=False)
        v = mst_arr[v_field].astype(np.int64, copy=False)

        out = np.empty((mst_arr.shape[0], 3), dtype=np.float64)
        out[:, 0] = u
        out[:, 1] = v

    # 2) Validar índices
    if u.min(initial=0) < 0 or v.min(initial=0) < 0 or u.max(initial=-1) >= n or v.max(initial=-1) >= n:
        raise IndexError("A MST contém índices fora do range da distance_matrix.")

    # 3) Substituir pesos pelo valor correto da matriz precomputed
    out[:, 2] = D[u, v].astype(np.float64, copy=False)

    return out


def add_mst_edges_to_metric_edges(
    metric_edges: np.ndarray,
    min_spanning_tree: np.ndarray,
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """
    Adiciona ao vetor metric_edges (bigger, smaller, dist) as arestas vindas da MST,
    APENAS se ainda não existirem em metric_edges. Sem repetições.

    Entradas esperadas
    ------------------
    metric_edges: ndarray (E, 3)
        Colunas: [bigger_idx, smaller_idx, dist]
        (triangular inferior: bigger = max(u,v), smaller = min(u,v))

    min_spanning_tree: ndarray (M, 3)
        Pode estar como:
          - [u, v, w] (qualquer ordem)
          - ou já [bigger, smaller, w]
        A função normaliza via max/min.

    n_nodes:
        Base para hashing dos pares (bigger, smaller).
        Se None, inferimos a partir dos maiores índices em metric_edges e MST.

    Retorna
    -------
    metric_edges_out: ndarray (E + added, 3) float64
        metric_edges original + novas arestas da MST que não existiam.
    """
    me = np.asarray(metric_edges)
    mst = np.asarray(min_spanning_tree)

    if me.ndim != 2 or me.shape[1] < 3:
        raise ValueError("metric_edges deve ser (E, 3) com [bigger, smaller, dist].")
    if mst.ndim != 2 or mst.shape[1] < 3:
        raise ValueError("min_spanning_tree deve ser (M, 3) com [u, v, weight].")

    # Força tipos/contiguidade
    me = np.ascontiguousarray(me, dtype=np.float64)
    mst = np.ascontiguousarray(mst, dtype=np.float64)

    me_b = me[:, 0].astype(np.int64, copy=False)
    me_s = me[:, 1].astype(np.int64, copy=False)

    mst_u = mst[:, 0].astype(np.int64, copy=False)
    mst_v = mst[:, 1].astype(np.int64, copy=False)
    mst_w = mst[:, 2].astype(np.float64, copy=False)

    # Normaliza MST -> triangular (bigger, smaller)
    mst_b = np.maximum(mst_u, mst_v)
    mst_s = np.minimum(mst_u, mst_v)

    # Base para chave única (bigger, smaller) -> bigger*base + smaller
    if n_nodes is None:
        max_idx = 0
        if me_b.size:
            max_idx = max(max_idx, int(me_b.max()))
        if me_s.size:
            max_idx = max(max_idx, int(me_s.max()))
        if mst_b.size:
            max_idx = max(max_idx, int(mst_b.max()))
        if mst_s.size:
            max_idx = max(max_idx, int(mst_s.max()))
        n_nodes = max_idx + 1

    base = int(n_nodes)
    if base <= 0:
        raise ValueError("n_nodes/base inválido.")

    # Chaves existentes
    me_key = me_b * base + me_s
    me_key_sorted = np.sort(me_key, kind="mergesort")

    # Chaves MST
    mst_key = mst_b * base + mst_s

    # Membership via searchsorted (mais rápido que np.isin em muitos cenários)
    pos = np.searchsorted(me_key_sorted, mst_key)
    exists = (pos < me_key_sorted.size) & (me_key_sorted[pos] == mst_key)

    # Filtra apenas as novas arestas
    add_mask = ~exists
    if not np.any(add_mask):
        return me  # nada a adicionar

    add_b = mst_b[add_mask].astype(np.float64, copy=False)
    add_s = mst_s[add_mask].astype(np.float64, copy=False)
    add_w = mst_w[add_mask]

    to_add = np.empty((add_w.size, 3), dtype=np.float64)
    to_add[:, 0] = add_b
    to_add[:, 1] = add_s
    to_add[:, 2] = add_w

    return np.vstack([me, to_add])
