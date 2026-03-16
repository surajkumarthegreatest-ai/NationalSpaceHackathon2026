import numpy as np

# --- CONSTANTS (KM / S) ---
MU_EARTH = 398600.4418
R_EARTH = 6378.137
J2 = 1.08262668e-3

def two_body_acceleration(r: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Vectorized two-body acceleration for (N, 3) batch."""
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = -mu * r / r_norm**3
    return np.nan_to_num(acc)

def j2_acceleration(r: np.ndarray, mu: float = MU_EARTH, j2: float = J2, re: float = R_EARTH) -> np.ndarray:
    """Vectorized J2 acceleration for (N, 3) batch."""
    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    r_sq = np.sum(r**2, axis=1, keepdims=True)
    r_norm = np.sqrt(r_sq)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        z_r_sq = (z / r_norm)**2
        factor = 1.5 * j2 * mu * (re**2) / (r_norm**5)
        common = 5 * z_r_sq - 1
        
        ax = factor * x * common
        ay = factor * y * common
        az = factor * z * (5 * z_r_sq - 3)
        
    return np.nan_to_num(np.column_stack((ax, ay, az)))

def total_acceleration(r: np.ndarray, use_j2: bool = False, mu: float = MU_EARTH) -> np.ndarray:
    """Batch acceleration logic."""
    a = two_body_acceleration(r, mu=mu)
    if use_j2:
        a += j2_acceleration(r, mu=mu)
    return a

def _derivative(state: np.ndarray, acc_func) -> np.ndarray:
    """Computes [v, a] for (N, 6) state batch."""
    # Slicing is a view, not a copy
    r = state[:, :3]
    v = state[:, 3:]
    a = acc_func(r)
    # Returns (N, 6)
    return np.column_stack((v, a))

def rk4_step(state: np.ndarray, dt: float, acc_func) -> np.ndarray:
    """Standard RK4 batch update."""
    k1 = _derivative(state, acc_func)
    k2 = _derivative(state + 0.5 * dt * k1, acc_func)
    k3 = _derivative(state + 0.5 * dt * k2, acc_func)
    k4 = _derivative(state + dt * k3, acc_func)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def propagate(state0: np.ndarray, dt: float, steps: int, use_j2: bool = False, mu: float = MU_EARTH) -> np.ndarray:
    """
    Propagates N satellites and returns ONLY the final state.
    RAM usage: O(N) instead of O(steps * N).
    """
    curr_state = np.asarray(state0, dtype=float)
    if curr_state.ndim == 1:
        curr_state = curr_state.reshape(1, -1)

    def acc_batch(r):
        return total_acceleration(r, use_j2=use_j2, mu=mu)

    # Hot loop: Memory stable
    for _ in range(steps):
        curr_state = rk4_step(curr_state, dt, acc_batch)

    return curr_state