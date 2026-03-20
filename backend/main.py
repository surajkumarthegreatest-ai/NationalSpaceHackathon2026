import os
import uvicorn
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Core Modules (Importing the raw math, not a wrapper)
from physics.propagator import rk4_step_hardened
from physics.maneuvers import ManeuverManager
from algorithms.conjunction import detect_conjunctions
from algorithms.scheduler import calculate_evasion_burns
from algorithms.optimization import calculate_optimal_evasion_batch

# --- DYNAMIC CONFIGURATION ---
FLEET_SIZE = int(os.getenv("FLEET_SIZE", 10000))
WET_MASS = float(os.getenv("WET_MASS", 550.0))
LATENCY_WINDOW_S = float(os.getenv("LATENCY_WINDOW_S", 15.0))
PROPAGATOR_STEPS = int(os.getenv("PROPAGATOR_STEPS", 3))
COLLISION_THRESHOLD_KM = float(os.getenv("COLLISION_THRESHOLD_KM", 0.5))

app = FastAPI(title="ACM: Autonomous Constellation Manager")
manager = ManeuverManager(n_satellites=FLEET_SIZE, initial_wet_mass=WET_MASS)

class TickRequest(BaseModel):
    current_time: float
    states: List[List[float]]

# --- THREAD-SAFE PHYSICS WORKER ---
def run_physics_batch(initial_states: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """
    Allocates the zero-allocation K-buffers exactly ONCE per thread execution.
    Creates a working copy to prevent time-travel data corruption of the present state.
    """
    state_copy = initial_states.copy()
    
    buffers = {
        'k1': np.empty_like(state_copy),
        'k2': np.empty_like(state_copy),
        'k3': np.empty_like(state_copy),
        'k4': np.empty_like(state_copy),
        'temp': np.empty_like(state_copy)
    }

    def acc_func(r_vecs):
        # Earth Gravitational Parameter (km^3/s^2)
        MU = 398600.4418 
        r_norm = np.linalg.norm(r_vecs, axis=1, keepdims=True)
        return -MU * r_vecs / (r_norm**3 + 1e-12)
        # Note: J2 perturbation math can be injected here seamlessly

    for _ in range(steps):
        rk4_step_hardened(state_copy, dt, acc_func, buffers)

    return state_copy


@app.post("/api/tick")
async def process_tick(payload: TickRequest):
    data = np.array(payload.states)
    
    if data.shape[0] > FLEET_SIZE:
        raise HTTPException(status_code=400, detail="Payload exceeds configured FLEET_SIZE.")

    is_debris = data[:, 1].astype(bool)
    current_states = data[:, 2:8] 

    # 1. Asynchronous Physics Offloading
    # We pass our local run_physics_batch to the thread instead of the missing wrapper
    dt_step = LATENCY_WINDOW_S / PROPAGATOR_STEPS
    future_states = await asyncio.to_thread(
        run_physics_batch, current_states, dt=dt_step, steps=PROPAGATOR_STEPS
    )
    
    # 2. Parameterized Conjunction Detection
    conjunction_pairs = detect_conjunctions(future_states[:, :3], threshold_km=COLLISION_THRESHOLD_KM)

    if not conjunction_pairs:
        return {"time": payload.current_time, "maneuvers": []}

    threat_sat_indices = []
    threat_debris_indices = []

    for i, j in conjunction_pairs:
        if not is_debris[i]:
            threat_sat_indices.append(i); threat_debris_indices.append(j)
        elif not is_debris[j]:
            threat_sat_indices.append(j); threat_debris_indices.append(i)

    if threat_sat_indices:
        _, first_occ = np.unique(threat_sat_indices, return_index=True)
        sat_indices = np.array(threat_sat_indices)[first_occ]
        deb_indices = np.array(threat_debris_indices)[first_occ]
        
        # 3. Dynamic Optimization
        dv_mags = calculate_optimal_evasion_batch(
            future_states[sat_indices], 
            future_states[deb_indices],
            min_miss_km=COLLISION_THRESHOLD_KM
        )

        # 4. Calculate Vectors
        delta_vs = calculate_evasion_burns(
            future_states[sat_indices], 
            future_states[deb_indices],
            dv_mag_km_s=dv_mags 
        )

        # 5. Atomic Execution
        _, valid_mask = manager.execute_batch(
            current_states, sat_indices, delta_vs, payload.current_time
        )

        # 6. JSON Formatting
        maneuvers = [
            {"satellite_id": int(sat_indices[idx]), "burn_vector_eci": delta_vs[idx].tolist()}
            for idx in range(len(sat_indices)) if valid_mask[idx]
        ]

        return {"time": payload.current_time, "maneuvers": maneuvers}

    return {"time": payload.current_time, "maneuvers": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)