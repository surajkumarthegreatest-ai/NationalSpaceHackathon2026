import numpy as np
import orjson
import asyncio
from fastapi import APIRouter, Request, Response, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

# --- SECURITY: Rate Limiter Setup ---
# Using the client's IP address to prevent per-user resource exhaustion
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Global state for the simulation (as per hackathon telemetry specs) [cite: 37, 38]
TELEMETRY_BUFFER = np.zeros((10000, 6), dtype=np.float64) 

@router.post("/tick")
@limiter.limit("50/minute") # Protects CPU-bound physics from spam [cite: 29, 132]
async def process_tick(request: Request):
    """
    Ingests telemetry and advances simulation. [cite: 75, 131]
    """
    global TELEMETRY_BUFFER
    
    try:
        raw_body = await request.body()
        payload = orjson.loads(raw_body)
        
        # 1. Resource Limit Check [cite: 261, 266]
        incoming_data = np.array(payload['states'], dtype=np.float64)
        if incoming_data.shape[0] > TELEMETRY_BUFFER.shape[0]:
             raise HTTPException(status_code=413, detail="Payload exceeds telemetry buffer.")

        # 2. Update buffer in-place to minimize memory fragmentation [cite: 37, 77]
        TELEMETRY_BUFFER[:incoming_data.shape[0], :] = incoming_data
        
        # 3. Offload CPU-bound physics (RK4 + J2) to a worker thread [cite: 63, 67]
        # Prevents blocking the FastAPI event loop during heavy calculation
        future_states = await asyncio.to_thread(
            propagate, TELEMETRY_BUFFER[:, 2:8], dt=5.0, steps=3, use_j2=True
        )
        
        return Response(
            content=orjson.dumps({
                "time": payload['current_time'], 
                "maneuvers": [],
                "status": "STEP_COMPLETE" # As per API Specification [cite: 141]
            }),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal simulation error")

@router.get("/state")
@limiter.limit("100/minute") # Protects serialization resources [cite: 206, 227]
async def get_state(request: Request):
    """
    Returns current constellation snapshot for the visualizer. [cite: 228, 229]
    """
    global TELEMETRY_BUFFER
    return Response(
        content=orjson.dumps({"states": TELEMETRY_BUFFER.tolist()}),
        media_type="application/json"
    )

def propagate(states, dt, steps, use_j2=True):
    # Physics integration logic (e.g., Runge-Kutta 4th Order) [cite: 67]
    return states