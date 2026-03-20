import numpy as np
from fastapi import APIRouter, HTTPException
from api.schemas import TickRequest, TickResponse, Maneuver

# Internal Engine Imports
from physics.propagator import propagate
from physics.maneuvers import ManeuverManager
from algorithms.conjunction import detect_conjunctions
from algorithms.scheduler import calculate_evasion_burns
from algorithms.optimization import calculate_optimal_evasion_batch

router = APIRouter(prefix="/api")

# --- PERSISTENT STATE ---
# Initialized for the 10,000 object scale mandated by the Hackathon
manager = ManeuverManager(n_satellites=10000)

# Global cache for the "Orbital Insight" Visualizer
# Ensures the GET /state endpoint has the latest "Ground Truth" [cite: 8]
latest_telemetry = []

@router.post("/tick", response_model=TickResponse)
async def process_tick(payload: TickRequest):
    """
    Main Orchestration Endpoint:
    Processes 10,000 objects, predicts collisions, and returns optimized maneuvers.
    """
    global latest_telemetry
    
    try:
        # 1. Ingest & Cache Telemetry
        # We store the raw states immediately for the frontend visualizer [cite: 3, 4]
        latest_telemetry = payload.states
        data = np.array(payload.states)
        
        if data.shape[1] != 8:
            raise ValueError("Invalid state vector format. Expected [id, is_debris, rx, ry, rz, vx, vy, vz]")

        ids = data[:, 0].astype(int)
        is_debris = data[:, 1].astype(bool)
        current_states = data[:, 2:8]

        # 2. Beat the 10s Latency: Propagate 15s into the future
        # Uses Vectorized RK4 + J2 to find the projected collision reality
        future_states = propagate(current_states, dt=5.0, steps=3, use_j2=True)

        # 3. Spatial Search (O(N log N))
        # Identifies proximity pairs using the cKDTree engine
        conjunction_pairs = detect_conjunctions(future_states[:, :3], threshold_km=0.5)

        if not conjunction_pairs:
            return TickResponse(time=payload.current_time, maneuvers=[])

        # 4. Filter & Deduplicate Threats
        # Ensures each satellite only receives ONE optimized maneuver per tick
        threat_sat_idx = []
        threat_deb_idx = []
        
        for i, j in conjunction_pairs:
            if not is_debris[i]:
                threat_sat_idx.append(i); threat_deb_idx.append(j)
            elif not is_debris[j]:
                threat_sat_idx.append(j); threat_deb_idx.append(i)

        if not threat_sat_idx:
            return TickResponse(time=payload.current_time, maneuvers=[])

        # Deduplication: Prevents "Duplicate ID Annihilation" and 600s cooldown violations
        _, first_occ = np.unique(threat_sat_idx, return_index=True)
        sat_indices = np.array(threat_sat_idx)[first_occ]
        deb_indices = np.array(threat_deb_idx)[first_occ]

        # 5. Optimized Evasion Calculation (Vectorized)
        # Calculates the minimal Delta-V required based on future TCA geometry
        dv_mags = calculate_optimal_evasion_batch(
            future_states[sat_indices], 
            future_states[deb_indices]
        )
        
        # Calculate ECI directions from the optimized magnitudes
        delta_vs = calculate_evasion_burns(
            future_states[sat_indices], 
            future_states[deb_indices],
            dv_mag_km_s=dv_mags 
        )

        # 6. Atomic State Update
        # Updates fuel mass and sets the 600s thermal cooldown for maneuvered satellites
        _, valid_mask = manager.execute_batch(
            current_states, sat_indices, delta_vs, payload.current_time
        )

        # 7. Serialize Response
        # Filter only maneuvers that passed the ManeuverManager's validation
        final_maneuvers = [
            Maneuver(
                satellite_id=int(ids[sat_indices[i]]), 
                burn_vector_eci=delta_vs[i].tolist()
            )
            for i in range(len(sat_indices)) if valid_mask[i]
        ]

        return TickResponse(time=payload.current_time, maneuvers=final_maneuvers)

    except Exception as e:
        # Standardize error reporting for the Hackathon grading server
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/state")
async def get_current_state():
    """
    Visualizer Data Bridge:
    Provides the frontend with the latest telemetry cache to render the Ghost Trace. [cite: 36]
    """
    return {"states": latest_telemetry}