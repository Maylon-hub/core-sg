# CORE_SG
import numpy as np
from sklearn.metrics import pairwise_distances
from joblib import Memory
from utils import knn_from_precomputed,build_knng_vectors,mst_with_precomputed_distances,add_mst_edges_to_metric_edges

import hdbscan



def _hdbscan_core_sg(
    X,
    k_max=5,
    metric="minkowski",
    p=2,
    return_label_k_max=False,
    **kwargs
):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    knng_size = X.shape[0]
    # The Cython routines used require contiguous arrays
    if not X.flags["C_CONTIGUOUS"]:
        X = np.array(X, dtype=np.double, order="C")

    if metric == "minkowski":
        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        distance_matrix = pairwise_distances(X, metric="cosine", **kwargs)
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)
    

    idx_arr, distances_arr = knn_from_precomputed(distance_matrix, k=k_max)


    # Get distance to kth nearest neighbour
    core_distances = distances_arr[:, -1].copy(order="C")

    metric_edges, knng_to_insert = build_knng_vectors(idx_arr,distances_arr,knng_size, k_max)

    # Executa a Primeira Iteração do HDBSCNA com o pre-computed e retornando a MST
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=None,
        min_samples=k_max,
        metric='precomputed',
        p=p,
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        match_reference_implementation=True,
        **kwargs)
    clusterer.fit(distance_matrix)
    min_spanning_tree = clusterer._min_spanning_tree.copy()

    labels = None
    if return_label_k_max:
        labels = clusterer.labels_.copy()

    del clusterer

    min_spanning_tree = mst_with_precomputed_distances(min_spanning_tree, distance_matrix, copy= False)

    metric_edges = add_mst_edges_to_metric_edges(
        metric_edges=metric_edges,          # (E,3) [bigger, smaller, dist]
        min_spanning_tree=min_spanning_tree # (M,3) [u, v, weight] ou [bigger, smaller, weight]
    )

    core_sg = np.concatenate((knng_to_insert, min_spanning_tree), axis=0)
    del knng_to_insert


    return core_sg,labels

    


def mst_from_core_sg(X,mpts,core_sg,core_distances,metric_distances,gen_min_span_tree=False):

    def update_core_sg(core_sg,mpts):
        for node in core_sg:
            bigger_idx = int(node[0])
            smaller_idx= int(node[1])

            if bigger_idx < smaller_idx:
                bigger_idx = int(node[1])
                smaller_idx = int(node[0])
            
            node[2] = max(core_distances[int(node[0]),mpts-1] , core_distances[int(node[1]),mpts-1],metric_distances[bigger_idx][smaller_idx])


    # Update weights
    update_core_sg(core_sg,mpts)


    from .mst_kruskal import kruskal_mst
    min_spanning_tree  = kruskal_mst(core_sg, X.shape[0])
    min_spanning_tree =  min_spanning_tree[np.argsort(min_spanning_tree.T[2]),:]

    
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)
    

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def clustering_from_core_sg(    
    X,
    min_cluster_size,
    core_sg,
    core_distances,
    metric_distances,
    gen_min_span_tree=False,
    cluster_selection_epsilon=0.0,
    memory=Memory(None, verbose=0),
    max_cluster_size=0,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False):

    (single_linkage_tree, result_min_span_tree) = memory.cache(
                mst_from_core_sg
            )(X,min_cluster_size,core_sg,core_distances,metric_distances,gen_min_span_tree)

    return (
            _tree_to_labels(
                X,
                single_linkage_tree,
                min_cluster_size,
                cluster_selection_method,
                allow_single_cluster,
                match_reference_implementation,
                cluster_selection_epsilon,
                max_cluster_size,
            )
            + (result_min_span_tree,)
        )




