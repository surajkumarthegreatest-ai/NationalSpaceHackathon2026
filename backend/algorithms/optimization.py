import numpy as np

# --- MISSION CONSTANTS ---
MAX_DV_KM_S = 0.015

def calculate_optimal_evasion_batch(
    sat_states: np.ndarray, 
    debris_states: np.ndarray, 
    min_miss_km: float = 0.5
) -> np.ndarray:
    """
    Vectorized calculation of minimal evasion burns for N pairs at TCA.
    
    Args:
        sat_states: (N, 6) batch of satellite states [r, v]
        debris_states: (N, 6) batch of debris states [r, v]
        min_miss_km: Safety radius (km)
        
    Returns:
        dv_mag_batch: (N,) array of required Delta-V magnitudes in km/s
    """
    N = sat_states.shape[0]
    if N == 0:
        return np.array([])

    # 1. Calculate Relative Geometry
    # r_rel: (N, 3), v_rel: (N, 3)
    r_rel = debris_states[:, :3] - sat_states[:, :3]
    v_rel = debris_states[:, 3:] - sat_states[:, 3:]

    # 2. Compute Time of Closest Approach (TCA) for the batch
    # dot(r, v) using einsum: 'ni,ni->n'
    # dot(v, v) using einsum: 'ni,ni->n'
    rv_dot = np.einsum('ni,ni->n', r_rel, v_rel)
    vv_dot = np.einsum('ni,ni->n', v_rel, v_rel)

    # Use a safety mask to avoid division by zero (objects moving in parallel)
    valid_motion = vv_dot > 1e-9
    t_tca = np.zeros(N)
    
    # t_tca = - (r . v) / |v|^2
    # We only care about FUTURE collisions (t_tca > 0)
    t_tca[valid_motion] = -rv_dot[valid_motion] / vv_dot[valid_motion]
    t_tca = np.maximum(t_tca, 0.0)

    # 3. Compute Miss Distance at TCA
    # r_tca = r_rel + v_rel * t_tca
    # r_tca shape: (N, 3)
    r_tca = r_rel + v_rel * t_tca[:, np.newaxis]
    d_miss = np.linalg.norm(r_tca, axis=1)

    # 4. Calculate Required Displacement
    # If d_miss < min_miss_km, we need to nudge the satellite.
    needed_displacement = np.maximum(0.0, min_miss_km - d_miss)

    # 5. Translate Displacement to Delta-V
    # Logic: For short-term evasion, Delta_V is roughly Displacement / Time.
    # We apply a 20% engineering safety margin for J2/Numerical Drift.
    dv_required = np.zeros(N)
    
    # Only calculate burn if t_tca is significant and displacement is needed
    burn_mask = (t_tca > 1.0) & (needed_displacement > 0)
    dv_required[burn_mask] = (needed_displacement[burn_mask] / t_tca[burn_mask]) * 1.2

    # 6. Clamp to Physical Limit (15 m/s)
    return np.clip(dv_required, 0.0, MAX_DV_KM_S)
