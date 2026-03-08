from __future__ import annotations
from dataclasses import dataclass
import numpy as np



@dataclass(frozen=True)
class CoreSGValidationReport:
    n: int
    k_max: int
    ok: bool
    missing_in_core: int


def validate_mst_in_core_sg(
    core_sg: np.ndarray,
    mst_hdb: np.ndarray,
    n:int,
    k: int
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    # --- Comparação MST: arestas + pesos ---
    
    d_hdb = {}
    a = 0
    ok = True

    for core in core_sg:
        max_val = max(core[:2])
        min_val = min(core[:2])
        if max_val not in d_hdb:
            d_hdb[max_val] = {}
        d_hdb[max_val][min_val] = [core[2]]
    for hdb in mst_hdb:
        max_val = max(hdb[:2])
        min_val = min(hdb[:2])
        try:
            d_hdb[max_val][min_val].append(hdb[2])
        except:
            a += 1
            ok = False

    
    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k),
        ok=bool(ok),
        missing_in_core=int(a),
    )

def sort_mst(mst: np.ndarray) -> np.ndarray:
    """
    Normaliza a MST para o formato [menor_idx, maior_idx, distancia]
    e ordena por (distancia, maior_idx, menor_idx).
    """
    if mst.ndim != 2 or mst.shape[1] < 3:
        raise ValueError("mst deve ter shape (n_edges, 3)")

    u = mst[:, 0].astype(np.int64, copy=False)
    v = mst[:, 1].astype(np.int64, copy=False)
    w = mst[:, 2].astype(np.float64, copy=False)

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    mst_tmp = np.empty((mst.shape[0], 3), dtype=np.float64)
    mst_tmp[:, 0] = u_min
    mst_tmp[:, 1] = v_max
    mst_tmp[:, 2] = w

    order = np.lexsort((mst_tmp[:, 0], mst_tmp[:, 1], mst_tmp[:, 2]))
    mst_tmp = mst_tmp[order]

    return mst_tmp

def validate_mst_from_core_sg(
    mst_core: np.ndarray,
    mst_hdb: np.ndarray,
    n:int,
    k: int
) -> CoreSGValidationReport:
    """
    Valida Core-SG vs HDBSCAN referência (MST exata).
    - O Core-SG calcula pairwise distances internamente.
    - O HDBSCAN referência roda com metric="precomputed" sobre a mesma matriz D.
    - Compara MST de mutual reachability (arestas + pesos).

    Retorna CoreSGValidationReport.
    """
    # --- Comparação MST: arestas + pesos ---
    mst_core_copy = sort_mst(mst_core)
    mst_hdb_copy = sort_mst(mst_hdb)
    ok = True
    a = 0

    for index,(core,hdb) in enumerate(zip(mst_core_copy,mst_hdb_copy)):
        max_hdb,min_hdb,weight_hdb = int(max(hdb[:2])),int(min(hdb[:2])),hdb[2]
        max_core,min_core,weight_core = int(max(core[:2])),int(min(core[:2])),core[2]
        
        if max_hdb != max_core or min_core != min_hdb or abs(weight_hdb - weight_core) > 0.001:
            print(core)
            print(hdb)
            a += 1
            ok = False



    return CoreSGValidationReport(
        n=int(n),
        k_max=int(k),
        ok=bool(ok),
        missing_in_core=int(a),
    )