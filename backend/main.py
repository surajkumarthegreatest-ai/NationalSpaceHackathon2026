import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Core Modules
from physics.propagator import propagate
from physics.maneuvers import ManeuverManager
from algorithms.conjunction import detect_conjunctions
from algorithms.scheduler import calculate_evasion_burns

app = FastAPI(title="ACM: Autonomous Constellation Manager")

# INITIALIZATION: 
# The Hackathon spec guarantees IDs are contiguous integers from 0 to N-1.
# This allows O(1) direct indexing into our state manager.
manager = ManeuverManager(n_satellites=10000, initial_wet_mass=550.0)

class TickRequest(BaseModel):
    current_time: float
    states: List[List[float]] # Format: [id, is_debris, rx, ry, rz, vx, vy, vz]

@app.post("/api/tick")
async def process_tick(payload: TickRequest):
    # 1. Data Ingestion
    # data[:, 0] = ID, data[:, 1] = is_debris, data[:, 2:8] = State
    data = np.array(payload.states)
    is_debris = data[:, 1].astype(bool)
    current_states = data[:, 2:8] 

    # 2. Beat the Latency: Predict the "Collision Reality" at T_now + 15s
    # We propagate the entire fleet using our vectorized RK4 + J2 kernel.
    future_states = propagate(current_states, dt=5.0, steps=3, use_j2=True)
    
    # 3. Detect Conjunctions on the FUTURE states (O(N log N))
    # threshold_km=0.5 provides a safety buffer for the 15s window.
    conjunction_pairs = detect_conjunctions(future_states[:, :3], threshold_km=0.5)

    if not conjunction_pairs:
        return {"time": payload.current_time, "maneuvers": []}

    # 4. Filter & De-duplicate Threats
    threat_sat_indices = []
    threat_debris_indices = []

    for i, j in conjunction_pairs:
        # Assign 'i' as the evader if it's a satellite
        if not is_debris[i]:
            threat_sat_indices.append(i)
            threat_debris_indices.append(j)
        # Or assign 'j' as the evader if it's a satellite
        elif not is_debris[j]:
            threat_sat_indices.append(j)
            threat_debris_indices.append(i)

    if threat_sat_indices:
        # CRITICAL FIX: The Duplicate ID Annihilation
        # np.unique with return_index ensures we only take ONE burn per satellite.
        # This prevents the ManeuverManager from throwing a ValueError/Corruption.
        _, first_occurrence = np.unique(threat_sat_indices, return_index=True)
        
        final_sat_indices = np.array(threat_sat_indices)[first_occurrence]
        final_debris_indices = np.array(threat_debris_indices)[first_occurrence]
        
        # 5. Calculate Evasion Burns using FUTURE geometry
        # FIX: We pass future_states[idx] so the RTN frame matches the collision point.
        delta_vs = calculate_evasion_burns(
            future_states[final_sat_indices], 
            future_states[final_debris_indices],
            dv_mag_km_s=0.005 # 5 m/s nudge
        )

        # 6. Atomic Validation & Execution
        # We pass the satellite IDs (which match their index) to the manager.
        # This updates mass and thermal cooldowns globally.
        _, valid_mask = manager.execute_batch(
            current_states, final_sat_indices, delta_vs, payload.current_time
        )

        # 7. Format JSON Response
        maneuvers = []
        valid_ids = final_sat_indices[valid_mask]
        valid_burns = delta_vs[valid_mask]
        
        for idx in range(len(valid_ids)):
            maneuvers.append({
                "satellite_id": int(valid_ids[idx]),
                "burn_vector_eci": valid_burns[idx].tolist()
            })

        return {"time": payload.current_time, "maneuvers": maneuvers}

    return {"time": payload.current_time, "maneuvers": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
