from __future__ import annotations
import numpy as np


def knn_from_precomputed(D: np.ndarray, k: int, *, include_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrai k vizinhos mais próximos de cada ponto a partir de uma matriz NxN de distâncias.
    Retorna (idxs, dists) com shape (N, k), ordenados crescente por distância.
    """
    D = np.asarray(D)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[1] != n:
        raise ValueError("D deve ser NxN.")
    if not include_self:
        if k <= 0 or k > n - 1:
            raise ValueError("k inválido (excluindo self, precisa k <= N-1).")
        Dwork = D.copy()
        np.fill_diagonal(Dwork, np.inf)
    else:
        if k <= 0 or k > n:
            raise ValueError("k inválido (incluindo self, precisa k <= N).")
        Dwork = D

    # Seleciona k menores por linha sem ordenar tudo
    idx_part = np.argpartition(Dwork, kth=k - 1, axis=1)[:, :k]
    dist_part = np.take_along_axis(Dwork, idx_part, axis=1)

    # Ordena apenas os k selecionados
    order = np.argsort(dist_part, axis=1, kind="mergesort")
    idx = np.take_along_axis(idx_part, order, axis=1).astype(np.int64, copy=False)
    dist = np.take_along_axis(dist_part, order, axis=1).astype(np.float64, copy=False)
    return idx, dist