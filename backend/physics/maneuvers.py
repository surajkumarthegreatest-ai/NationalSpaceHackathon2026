import numpy as np

# --- MISSION CONSTANTS ---
G0_KM_S2 = 0.00980665 
ISP_SECONDS = 300.0   
MAX_DV_KM_S = 0.015   
COOLDOWN_SECONDS = 600.0 
M_DRY = 500.0 

class ManeuverManager:
    def __init__(self, n_satellites: int, initial_wet_mass: float = 550.0):
        self.n_satellites = n_satellites
        self.masses = np.full(n_satellites, initial_wet_mass, dtype=np.float64)
        self.last_burn_times = np.full(n_satellites, -np.inf, dtype=np.float64)

    def validate_and_calculate(self, ids: np.ndarray, delta_vs: np.ndarray, current_time: float):
        """Atomic validation for (N,) batch maneuvers."""
        # 0. Reject duplicate IDs in the same batch to prevent buffer corruption
        unique_ids, counts = np.unique(ids, return_counts=True)
        if np.any(counts > 1):
            raise ValueError("CRITICAL: Duplicate satellite IDs detected in a single maneuver batch.")

        # 1. Magnitude Check (Must be > 0 and <= MAX)
        dv_mags = np.linalg.norm(delta_vs, axis=-1)
        dv_ok = (dv_mags > 0.0) & (dv_mags <= MAX_DV_KM_S)

        # 2. Thermal Cooldown Check 
        time_since_last = current_time - self.last_burn_times[ids]
        cooldown_ok = time_since_last >= COOLDOWN_SECONDS

        # 3. Precise Fuel Check
        exponent = dv_mags / (ISP_SECONDS * G0_KM_S2)
        potential_burn = self.masses[ids] * (1 - np.exp(-exponent))
        fuel_ok = (self.masses[ids] - potential_burn) >= M_DRY

        valid_mask = dv_ok & cooldown_ok & fuel_ok
        return valid_mask, potential_burn

    def execute_batch(self, state: np.ndarray, ids: np.ndarray, delta_vs: np.ndarray, current_time: float):
        """Executes strictly validated maneuvers."""
        valid_mask, burn_masses = self.validate_and_calculate(ids, delta_vs, current_time)
        
        valid_ids = ids[valid_mask]
        valid_dvs = delta_vs[valid_mask]
        valid_burns = burn_masses[valid_mask]

        if len(valid_ids) > 0:
            # Update state, mass, and cooldown
            state[valid_ids, 3:] += valid_dvs
            self.masses[valid_ids] -= valid_burns
            self.last_burn_times[valid_ids] = current_time

        return state, valid_mask