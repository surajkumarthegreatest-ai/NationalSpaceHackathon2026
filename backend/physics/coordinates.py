import numpy as np
from typing import Tuple

def eci_to_rtn_matrix(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes RTN rotation matrices and validity mask.
    M[:, :, 0] = Radial, M[:, :, 1] = Transverse, M[:, :, 2] = Normal.
    """
    N = r.shape[0]
    M = np.full((N, 3, 3), np.nan)
    
    r_sq = np.sum(r**2, axis=1, keepdims=True)
    h_vecs = np.cross(r, v)
    h_sq = np.sum(h_vecs**2, axis=1, keepdims=True)
    
    valid_mask = ((r_sq > 0) & (h_sq > 0)).flatten()
    
    if not np.any(valid_mask):
        return M, valid_mask
    
    R_hat = r[valid_mask] / np.sqrt(r_sq[valid_mask])
    N_hat = h_vecs[valid_mask] / np.sqrt(h_sq[valid_mask])
    T_hat = np.cross(N_hat, R_hat)
    
    # Stack as columns: M = [R | T | N]
    M[valid_mask] = np.stack((R_hat, T_hat, N_hat), axis=2)
    
    return M, valid_mask

def eci_to_rtn(M: np.ndarray, vec_eci: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    V_rtn = M^T * V_eci
    Corrected einsum: 'nij,nj->ni' sums over columns j of M_trans.
    """
    results = np.full_like(vec_eci, np.nan)
    if np.any(mask):
        # M_trans shape is (N, 3, 3). We multiply by vec (N, 3).
        # We sum over the column index 'j' of the matrix.
        M_trans = M[mask].transpose(0, 2, 1)
        results[mask] = np.einsum('nij,nj->ni', M_trans, vec_eci[mask])
    return results

def rtn_to_eci(M: np.ndarray, vec_rtn: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    V_eci = M * V_rtn
    Corrected einsum: 'nij,nj->ni' sums over columns j of M.
    """
    results = np.full_like(vec_rtn, np.nan)
    if np.any(mask):
        results[mask] = np.einsum('nij,nj->ni', M[mask], vec_rtn[mask])
    return results