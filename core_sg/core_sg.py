from __future__ import annotations
import numpy as np
import hdbscan
from sklearn.metrics import pairwise_distances

from .knn import knn_from_precomputed
from .edges import build_knng_vectors, add_mst_edges_to_metric_edges
from .mst_kruskal import kruskal_mst
from .reweight import reweight_core_sg_mutual_reachability,sort_core_sg


def hdbscan_reference_mst_original_distance(D: np.ndarray, **kwargs) -> np.ndarray:
    # min_samples=1 => mutual reachability == distância original
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,          # qualquer >=2 (não importa, você só quer a MST)
        min_samples=15,
        metric="precomputed",
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        **kwargs,
    )
    clusterer.fit(D)
    return np.asarray(clusterer._min_spanning_tree, dtype=np.float64)  # pesos ~ D[u,v]


def build_core_sg_from_data(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    pairwise_dtype=np.float64,
    test_only: bool = False,
    **hdbscan_kwargs,
):
    """
    Recebe X (n, d), calcula pairwise distances D (n, n) internamente e constrói:
      - core distances
      - kNNG vetorizado
      - metric_edges
      - (opcional) inclui arestas da MST da distância original no metric_edges e no core_sg

    Retorna:
      core_sg, metric_edges, core_k, mst_orig, D
    """
    X = np.asarray(X)
    n = X.shape[0]

    if n <= 1:
        raise ValueError("X precisa ter ao menos 2 pontos.")
    if k_max <= 0 or k_max >= n:
        raise ValueError("k_max inválido (precisa 1 <= k_max <= n-1).")
    if k_max < 2:
        raise ValueError("k_max deve ser >= 2 para reproduzir min_samples do HDBSCAN.")

    # ---- pairwise distances dentro da função ----
    if metric == "minkowski":
        D = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        D = pairwise_distances(X, metric="cosine")
    else:
        D = pairwise_distances(X, metric=metric)

    D = np.ascontiguousarray(D, dtype=pairwise_dtype)
    np.fill_diagonal(D, 0.0)

    if test_only:
        D = D.round(4)

    # ------------------------------------------------------------------
    # Separação correta dos papéis:
    # - min_samples_k: parâmetro do HDBSCAN
    # - graph_knn_k: número de vizinhos no kNN graph do Core-SG
    # - core_k: core-distance compatível com HDBSCAN
    # ------------------------------------------------------------------
    min_samples_k = k_max
    graph_knn_k = min_samples_k

    # kNN graph do Core-SG usa k original
    idxs_graph, dists_graph = knn_from_precomputed(
        D,
        k=graph_knn_k,
        include_self=False,
    )

    metric_edges, knng_to_insert = build_knng_vectors(
        idxs_graph,
        dists_graph,
        knng_size=n,
        k_max=graph_knn_k,
    )


    # min_samples conta o próprio ponto, então com diagonal 0
    # o índice correto é min_samples_k - 1
    #core_k = np.partition(D, kth=min_samples_k - 1, axis=1)[:, min_samples_k - 1]
    #core_k = np.ascontiguousarray(core_k, dtype=np.float64)

    core_k_list = np.partition(D, kth=min_samples_k - 1, axis=1)[:, :min_samples_k]
    core_k_list = np.sort(core_k_list, axis=1)



    # MST da distância original, não da mutual reachability com k_max
    mst_orig = hdbscan_reference_mst_original_distance(D, **hdbscan_kwargs)

    u = mst_orig[:, 0].astype(np.int64, copy=False)
    v = mst_orig[:, 1].astype(np.int64, copy=False)

    # metric_edges deve sempre guardar a distância original D[u,v]
    mst_for_metric_edges = np.empty((mst_orig.shape[0], 3), dtype=np.float64)
    mst_for_metric_edges[:, 0] = u
    mst_for_metric_edges[:, 1] = v
    mst_for_metric_edges[:, 2] = D[u, v]

    metric_edges = add_mst_edges_to_metric_edges(
        metric_edges,
        mst_for_metric_edges,
        n_nodes=n,
    )

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    mst_tmp = np.empty((mst_orig.shape[0], 3), dtype=np.float64)
    mst_tmp[:, 0] = u_min
    mst_tmp[:, 1] = v_max
    mst_tmp[:, 2] = -1.0  # placeholder

    core_sg = np.vstack([knng_to_insert, mst_tmp])
    core_sg = sort_core_sg(core_sg)

    return core_sg, metric_edges, core_k_list, D


def mst_from_core_sg(
    core_sg: np.ndarray,
    metric_edges: np.ndarray,
    core_k_list: np.ndarray,
    n_nodes: int,
    k: int
):
    """
    Reweight -> Kruskal -> label() (HDBSCAN internal) => single linkage tree.
    Retorna:
      mst_arr: (n-1,3) [u,v,w] ordenado por w
      single_linkage_tree: output do label()
    """

    if k <= 0 or k >= n_nodes:
        raise ValueError("k_max inválido (precisa 1 <= k_max <= n-1).")
    if k < 2:
        raise ValueError("k_max deve ser >= 2 para reproduzir min_samples do HDBSCAN.")

    core_k = core_k_list[:, k - 1]
    core_k = np.ascontiguousarray(core_k, dtype=np.float64)
    weighted = reweight_core_sg_mutual_reachability(
        core_sg=core_sg,
        core_k=core_k,
        metric_edges=metric_edges,
        n_nodes=n_nodes,
    )


    mst_rec = kruskal_mst(weighted, n_nodes=n_nodes)
    mst_arr = np.column_stack([mst_rec.u, mst_rec.v, mst_rec.distance]).astype(np.float64, copy=False)
    mst_arr = mst_arr[np.argsort(mst_arr[:, 2], kind="mergesort")]


    return mst_arr