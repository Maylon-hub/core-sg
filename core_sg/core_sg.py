from __future__ import annotations
import numpy as np
import hdbscan
from sklearn.metrics import pairwise_distances
from time import time
import pandas as pd
from typing import Any
from warnings import warn


from .knn import knn_from_precomputed
from .edges import build_knng_vectors, add_mst_edges_to_metric_edges
from .mst_kruskal import kruskal_mst
from .reweight import reweight_core_sg_mutual_reachability,sort_core_sg

from hdbscan.hdbscan_ import _tree_to_labels
from hdbscan._hdbscan_linkage import label
from hdbscan.plots import MinimumSpanningTree
from hdbscan.plots import SingleLinkageTree
from hdbscan.plots import CondensedTree



def hdbscan_reference_mst_original_distance(D: np.ndarray,k_max: int) -> np.ndarray:
    # min_samples=1 => mutual reachability == distância original
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=k_max,          # qualquer >=2 (não importa, você só quer a MST)
        min_samples=k_max,
        metric="precomputed",
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        match_reference_implementation=True
    )
    clusterer.fit(D)
    return clusterer,np.asarray(clusterer._min_spanning_tree, dtype=np.float64)  # pesos ~ D[u,v]


def build_core_sg_from_data(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    pairwise_dtype=np.float64,
    test_only: bool = False,
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



    # MST da mutual reachability com k_max
    hdb_obj,mst_orig = hdbscan_reference_mst_original_distance(D,k_max)

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

    return core_sg, metric_edges, core_k_list, D,hdb_obj


def mst_from_core_sg(
    core_sg: np.ndarray,
    metric_edges: np.ndarray,
    core_k_list: np.ndarray,
    n_nodes: int,
    k: int
):
    """
    Reweight -> Kruskal -> 
    Retorna:
      mst_arr: (n-1,3) [u,v,w] ordenado por w
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

def core_sg_mutual_reachability_distance( 
    core_sg: np.ndarray,
    metric_edges: np.ndarray,
    core_k_list: np.ndarray,
    n_nodes: int,
    k: int
):
    """
    Reweight CoreSG 
    Retorna:
      core_sg: (n-1,3) [u,v,w] ordenado por w
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

    return weighted


def tree_to_labels(
    obj: "CoreSG",
    single_linkage_tree: np.ndarray,
    min_spanning_tree: np.ndarray,
):
    """
    Convert a single linkage tree into the standard HDBSCAN outputs.

    Only the parameters explicitly provided in `obj.hdbscan_kwargs` and
    supported by `_tree_to_labels` are forwarded. Any missing parameter
    falls back to the default defined by the internal HDBSCAN function.

    Parameters
    ----------
    obj : CoreSG
        Fitted CoreSG instance containing the pairwise distance matrix (`_D`)
        and the keyword arguments originally provided at initialization.
    single_linkage_tree : np.ndarray
        Single linkage hierarchy built from the minimum spanning tree.
    min_spanning_tree : np.ndarray
        Minimum spanning tree associated with the hierarchy.

    Returns
    -------
    tuple
        A tuple with:
        (
            labels,
            probabilities,
            cluster_persistence,
            condensed_tree,
            single_linkage_tree,
            min_spanning_tree,
        )
    """

    tree_kwargs = obj._get_tree_to_labels_kwargs()

    return (
        _tree_to_labels(
            obj._D,
            single_linkage_tree,
            **tree_kwargs,
        )
        + (min_spanning_tree,)
    )


class CoreSG:
    """
    Core-SG wrapper for reusing HDBSCAN computations across different values of k.

    The class builds the Core-SG once for a reference `k_max`, stores the
    corresponding HDBSCAN outputs, and allows reconstructing the MST and the
    hierarchy for other values `k <= k_max`.

    Notes
    -----
    - All `hdbscan_kwargs` are forwarded to `build_core_sg_from_data(...)`.
    - Only a filtered subset of these arguments is forwarded to
      `hdbscan.hdbscan_._tree_to_labels(...)`.
    - This implementation depends on private functions from `hdbscan`.
    """

    _TREE_TO_LABELS_KEYS = {
        "cluster_selection_method",
        "allow_single_cluster",
        "match_reference_implementation",
        "cluster_selection_epsilon",
        "cluster_selection_persistence",
        "max_cluster_size",
        "cluster_selection_epsilon_max",
    }

    def __init__(
        self,
        metric: str = "euclidean",
        p: int = 2,
        debug: bool = False,
        **hdbscan_kwargs: Any,
    ) -> None:
        """
        Initialize the CoreSG object.

        Parameters
        ----------
        metric : str, default="euclidean"
            Distance metric used to build the pairwise distances / Core-SG.
        p : int, default=2
            Power parameter for metrics such as Minkowski.
        debug : bool, default=False
            If True, prints execution times for the main steps.
        **hdbscan_kwargs : Any
            Extra keyword arguments passed to `build_core_sg_from_data(...)`.
            A filtered subset is also reused in `_tree_to_labels(...)`.
        """
        self.metric = metric
        self.p = p
        self.debug = debug
        self.hdbscan_kwargs = dict(hdbscan_kwargs)

        # Global fit metadata
        self.n = None
        self.k_max = None
        self._raw_data = None

        # Cached artifacts for k_max
        self._condensed_tree_k_max = None
        self.labels_k_max = None
        self.probabilities_k_max = None
        self.cluster_persistence_k_max = None
        self._single_linkage_tree_k_max = None
        self._min_spanning_tree_k_max = None

        # Shared Core-SG structures
        self._core_sg = None
        self._metric_edges = None
        self._core_k_list = None
        self._D = None

        # Current artifacts for an extracted k
        self.labels_ = None
        self.probabilities_ = None
        self.cluster_persistence_ = None
        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None

    def get_core_sg_mutual_reachability_distance(self,k: int):
        return core_sg_mutual_reachability_distance( 
            self._core_sg,
            self._metric_edges,
            self._core_k_list,
            self.n,
            k
        )
    def get_core_distance(self,k):
        core_k = self._core_k_list[:, k - 1]
        core_k = np.ascontiguousarray(core_k, dtype=np.float64)
        return core_k
    
    def _get_tree_to_labels_kwargs(self) -> dict[str, Any]:
        """
        Extract only the keyword arguments supported by `_tree_to_labels`.

        Returns
        -------
        dict[str, Any]
            Dictionary containing only the supported keys that were explicitly
            provided and whose value is not None.
        """
        return {
            key: value
            for key, value in self.hdbscan_kwargs.items()
            if key in self._TREE_TO_LABELS_KEYS and value is not None
        }

    def _store_k_max_outputs(self, hdb_obj: Any) -> None:
        """
        Store the HDBSCAN outputs obtained during the reference fit at k_max.

        Parameters
        ----------
        hdb_obj : Any
            HDBSCAN-like fitted object returned by `build_core_sg_from_data`.
        """
        self.condensed_tree_k_max_ = hdb_obj._condensed_tree
        self.labels_k_max = hdb_obj.labels_
        self.probabilities_k_max = hdb_obj.probabilities_
        self.cluster_persistence_k_max = hdb_obj.cluster_persistence_
        self.single_linkage_tree_k_max_ = hdb_obj._single_linkage_tree
        self.minimum_spanning_tree_k_max_ = hdb_obj._min_spanning_tree


    @staticmethod
    def _mst_to_dataframe(mst: np.ndarray) -> pd.DataFrame:
        """
        Convert an MST array into a typed pandas DataFrame.

        Parameters
        ----------
        mst : np.ndarray
            MST stored as an array with columns [to, from, weight].

        Returns
        -------
        pd.DataFrame
            DataFrame with typed columns: `to`, `from`, `weight`.
        """
        return pd.DataFrame(
            mst,
            columns=["to", "from", "weight"],
        ).astype(
            {
                "to": "int64",
                "from": "int64",
                "weight": "float64",
            }
        )
   
    @property
    def condensed_tree_(self):
        """
        Return the current extracted condensed tree wrapped as an HDBSCAN object.

        Returns
        -------
        CondensedTree
            Wrapped condensed tree for the current extracted hierarchy.

        Raises
        ------
        AttributeError
            If no current condensed tree is available.
        """

        if self._condensed_tree is not None:
            return CondensedTree(self._condensed_tree, self.labels_)

        raise AttributeError(
            "No condensed tree was generated for the current k; "
            "try running extract_hierarchy_from_core_sg first."
        )

    @condensed_tree_.setter
    def condensed_tree_(self, value):
        self._condensed_tree = value

    @property
    def condensed_tree_k_max_(self):
        """
        Return the fit-time condensed tree wrapped as an HDBSCAN object.

        Returns
        -------
        CondensedTree
            Wrapped condensed tree saved during `fit`.

        Raises
        ------
        AttributeError
            If no fit-time condensed tree is available.
        """
        

        if self._condensed_tree_k_max is not None:
            return CondensedTree(self._condensed_tree_k_max, self.labels_k_max)

        raise AttributeError(
            "No condensed tree was saved from fit; try running fit first."
        )

    @condensed_tree_k_max_.setter
    def condensed_tree_k_max_(self, value):
        self._condensed_tree_k_max = value

    @property
    def single_linkage_tree_(self):
        """
        Return the current extracted single linkage tree wrapped as an HDBSCAN object.

        Returns
        -------
        SingleLinkageTree
            Wrapped single linkage tree for the current extracted hierarchy.

        Raises
        ------
        AttributeError
            If no current single linkage tree is available.
        """


        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)

        raise AttributeError(
            "No single linkage tree was generated for the current k; "
            "try running extract_hierarchy_from_core_sg first."
        )

    @single_linkage_tree_.setter
    def single_linkage_tree_(self, value):
        self._single_linkage_tree = value

    @property
    def single_linkage_tree_k_max_(self):
        """
        Return the fit-time single linkage tree wrapped as an HDBSCAN object.

        Returns
        -------
        SingleLinkageTree
            Wrapped single linkage tree saved during `fit`.

        Raises
        ------
        AttributeError
            If no fit-time single linkage tree is available.
        """

        if self._single_linkage_tree_k_max is not None:
            return SingleLinkageTree(self._single_linkage_tree_k_max)

        raise AttributeError(
            "No single linkage tree was saved from fit; try running fit first."
        )

    @single_linkage_tree_k_max_.setter
    def single_linkage_tree_k_max_(self, value):
        self._single_linkage_tree_k_max = value

    @property
    def minimum_spanning_tree_(self):
        """
        Return the current extracted MST wrapped as an HDBSCAN object.

        Returns
        -------
        MinimumSpanningTree or None
            Wrapped MST for the current extracted hierarchy. If no raw feature
            data is available, returns None and emits a warning.

        Raises
        ------
        AttributeError
            If no current MST is available.
        """

        if self._min_spanning_tree is None:
            raise AttributeError(
                "No minimum spanning tree was generated for the current k; "
                "try running extract_hierarchy_from_core_sg first."
            )

        if self._raw_data is not None:
            return MinimumSpanningTree(self._min_spanning_tree, self._raw_data)

        warn(
            "No raw data is available; this may be due to using a "
            "precomputed metric matrix. No minimum spanning tree object "
            "will be provided without raw data."
        )
        return None

    @minimum_spanning_tree_.setter
    def minimum_spanning_tree_(self, value):
        self._min_spanning_tree = value

    @property
    def minimum_spanning_tree_k_max_(self):
        """
        Return the fit-time MST wrapped as an HDBSCAN object.

        Returns
        -------
        MinimumSpanningTree or None
            Wrapped MST saved during `fit`. If no raw feature data is
            available, returns None and emits a warning.

        Raises
        ------
        AttributeError
            If no fit-time MST is available.
        """

        if self._min_spanning_tree_k_max is None:
            raise AttributeError(
                "No minimum spanning tree was saved from fit; "
                "try running fit first."
            )

        if self._raw_data is not None:
            return MinimumSpanningTree(self._min_spanning_tree_k_max, self._raw_data)

        warn(
            "No raw data is available; this may be due to using a "
            "precomputed metric matrix. No minimum spanning tree object "
            "will be provided without raw data."
        )
        return None

    @minimum_spanning_tree_k_max_.setter
    def minimum_spanning_tree_k_max_(self, value):
        self._min_spanning_tree_k_max = value

    def get_fitted_hdbscan_objects(self, wrapped: bool = True) -> dict[str, Any]:
        """
        Return the HDBSCAN artifacts saved during the Core-SG `fit` at `k_max`.

        Parameters
        ----------
        wrapped : bool, default=True
            If True, returns the tree artifacts wrapped using the same object
            types exposed by HDBSCAN (`CondensedTree`, `SingleLinkageTree`,
            `MinimumSpanningTree` when possible). If False, returns the raw
            internal arrays.

        Returns
        -------
        dict[str, Any]
            Dictionary with the artifacts saved during `fit`, namely:
            `labels_`, `probabilities_`, `cluster_persistence_`,
            `condensed_tree_`, `single_linkage_tree_`, and
            `minimum_spanning_tree_`.

        Raises
        ------
        AttributeError
            If the object has not been fitted yet.
        """
        if self.k_max is None:
            raise AttributeError("CoreSG is not fitted yet. Run fit first.")

        if wrapped:
            return {
                "labels_": self.labels_k_max,
                "probabilities_": self.probabilities_k_max,
                "cluster_persistence_": self.cluster_persistence_k_max,
                "condensed_tree_": self.condensed_tree_k_max_,
                "single_linkage_tree_": self.single_linkage_tree_k_max_,
                "minimum_spanning_tree_": self.minimum_spanning_tree_k_max_,
            }


        return {
            "labels_": self.labels_k_max,
            "probabilities_": self.probabilities_k_max,
            "cluster_persistence_": self.cluster_persistence_k_max,
            "condensed_tree_": self._condensed_tree_k_max,
            "single_linkage_tree_": self._single_linkage_tree_k_max,
            "minimum_spanning_tree_": self._min_spanning_tree_k_max,
        }

    def fit(self, X: np.ndarray, k_max: int,test_only: bool = False) -> "CoreSG":
        """
        Build the Core-SG for a reference value `k_max`.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        k_max : int
            Maximum neighborhood size used to build the Core-SG.

        Returns
        -------
        CoreSG
            The fitted instance itself.
        """
        self.n = X.shape[0]
        self.k_max = k_max

        # Segue a lógica do HDBSCAN: o wrapper do MST depende dos dados crus.
        self._raw_data = X

        t0 = time()
        (
            self._core_sg,
            self._metric_edges,
            self._core_k_list,
            self._D,
            hdb_obj,
        ) = build_core_sg_from_data(
            X,
            k_max=k_max,
            metric=self.metric,
            p=self.p,
            test_only=test_only
        )
        t1 = time()

        if self.debug:
            print(f"Core-SG build done in {t1 - t0:.2f}s")

        self._store_k_max_outputs(hdb_obj)
        return self

    def extract_mst_from_core_sg(self, k: int, toDF: bool = False):
        """
        Extract the minimum spanning tree for a given `k` from the Core-SG.

        Parameters
        ----------
        k : int
            Neighborhood size for which the MRD-based MST should be extracted.
        toDF : bool, default=False
            If True, returns the MST as a pandas DataFrame. Otherwise returns
            the raw NumPy array.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The extracted MST, either as a NumPy array or as a DataFrame.
        """
        if self.k_max == k:
            if toDF:
                return self._mst_to_dataframe(self._min_spanning_tree_k_max)
            return self._min_spanning_tree_k_max

        t0 = time()
        mst_core = mst_from_core_sg(
            core_sg=self._core_sg,
            metric_edges=self._metric_edges,
            core_k_list=self._core_k_list,
            n_nodes=self.n,
            k=k,
        )
        t1 = time()

        if self.debug:
            print(f"Core-SG MST K = {k} (Kruskal) done in {t1 - t0:.2f}s")

        if toDF:
            return self._mst_to_dataframe(mst_core)

        return mst_core

    def extract_hierarchy_from_core_sg(self, k: int) -> None:
        """
        Reconstruct the HDBSCAN hierarchy for a given `k` from the Core-SG.

        This method computes the MST for the requested `k`, converts it into
        a single linkage tree, and runs the HDBSCAN tree post-processing to
        populate the current attributes:
        `labels_`, `probabilities_`, `cluster_persistence_`,
        `condensed_tree_`, `single_linkage_tree_`, `minimum_spanning_tree_`.

        Parameters
        ----------
        k : int
            Neighborhood size for which the hierarchy should be extracted.

        Returns
        -------
        None
            The method updates the instance attributes in place.
        """


        if self.k_max == k:
            (
            self.labels_,
            self.probabilities_,
            self.cluster_persistence_,
            condensed_tree,
            single_linkage_tree,
            min_spanning_tree,
            ) = self.get_fitted_hdbscan_objects(wrapped=False).values()

            self._condensed_tree = condensed_tree
            self._single_linkage_tree = single_linkage_tree
            self._min_spanning_tree = min_spanning_tree

            return None

        min_spanning_tree = self.extract_mst_from_core_sg(k)
        single_linkage_tree = label(min_spanning_tree)

        t0 = time()
        (
            self.labels_,
            self.probabilities_,
            self.cluster_persistence_,
            condensed_tree,
            single_linkage_tree,
            min_spanning_tree,
        ) = tree_to_labels(self, single_linkage_tree, min_spanning_tree)
        t1 = time()

        self._condensed_tree = condensed_tree
        self._single_linkage_tree = single_linkage_tree
        self._min_spanning_tree = min_spanning_tree

        if self.debug:
            print(f"FOSC K = {k} done in {t1 - t0:.2f}s")

        return None

        
