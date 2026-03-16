import numpy as np
from physics.coordinates import eci_to_rtn_matrix

def calculate_evasion_burns(
    sat_states: np.ndarray, 
    debris_states: np.ndarray, 
    dv_mag_km_s: float = 0.005
) -> np.ndarray:
    """
    Vectorized evasion burn calculation for N simultaneous conjunctions.
    
    Args:
        sat_states: (N, 6) batch of satellite states [r, v]
        debris_states: (N, 6) batch of debris states [r, v]
        dv_mag_km_s: Magnitude of the nudge (default 5 m/s)
        
    Returns:
        dv_eci_batch: (N, 3) batch of ECI burn vectors
    """
    N = sat_states.shape[0]
    if N == 0:
        return np.zeros((0, 3))

    # 1. Extract Position and Velocity for Matrix Generation
    r_sat = sat_states[:, :3]
    v_sat = sat_states[:, 3:]
    
    # 2. Generate RTN Transformation Matrices (N, 3, 3)
    # This uses our O(1) vectorized kernel from coordinates.py
    M_batch, mask = eci_to_rtn_matrix(r_sat, v_sat)
    
    # 3. Determine Geometry: Relative position in ECI
    # rel_pos shape: (N, 3)
    rel_pos_eci = debris_states[:, :3] - r_sat
    
    # 4. Project Relative Position onto the Transverse (T) Axis
    # T_hat is the middle column of our [R|T|N] matrix
    T_hat = M_batch[:, :, 1]
    
    # Dot product for each row: (N, 3) * (N, 3) summed across axis 1 -> (N,)
    # If positive, debris is ahead of the satellite in its path.
    is_debris_ahead = np.einsum('ni,ni->n', rel_pos_eci, T_hat) > 0
    
    # 5. Build RTN Burn Vectors (N, 3)
    # Logic: Apply Transverse burn to shift the Time of Closest Approach.
    dv_rtn_batch = np.zeros((N, 3))
    
    # Direction logic: If debris is ahead, nudge retrograde (-T) to create gap.
    # Note: Short-term (15s) Transverse displacement is the goal here.
    dv_rtn_batch[:, 1] = np.where(is_debris_ahead, -dv_mag_km_s, dv_mag_km_s)
    
    # 6. Transform RTN Batch to ECI Batch
    # V_eci = M * V_rtn using our optimized einsum string 'nij,nj->ni'
    dv_eci_batch = np.einsum('nij,nj->ni', M_batch, dv_rtn_batch)
    
    # Apply mask: if matrix was invalid (NaN), return 0 burn for that satellite
    return np.nan_to_num(dv_eci_batch)
