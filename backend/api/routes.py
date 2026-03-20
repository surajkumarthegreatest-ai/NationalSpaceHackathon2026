import numpy as np
import asyncio
import orjson
from fastapi import APIRouter, Request, Response

router = APIRouter(prefix="/api")

# --- STATIC MEMORY POOL ---
# Pre-allocated once. We use slice assignment [:] to update.
TELEMETRY_BUFFER = np.zeros((10000, 8), dtype=np.float64)

@router.post("/tick")
async def process_tick(request: Request):
    global TELEMETRY_BUFFER
    
    # PRODUCTION FIX: Rapid deserialization + Thread Offloading
    raw_body = await request.body()
    payload = orjson.loads(raw_body)
    
    # Convert and update global buffer in-place
    incoming_data = np.array(payload['states'], dtype=np.float64)
    TELEMETRY_BUFFER[:incoming_data.shape[0], :] = incoming_data
    
    # Offload CPU-bound physics to a worker thread to keep the loop free
    future_states = await asyncio.to_thread(
        propagate, TELEMETRY_BUFFER[:, 2:8], dt=5.0, steps=3, use_j2=True
    )

    # ... (Maneuver & Logic) ...

    return Response(
        content=orjson.dumps({"time": payload['current_time'], "maneuvers": []}),
        media_type="application/json"
    )

@router.get("/state")
async def get_current_state():
    """Stream telemetry directly to the UI using ORJSON's NumPy support."""
    return Response(
        content=orjson.dumps(
            {"states": TELEMETRY_BUFFER}, 
            option=orjson.OPT_SERIALIZE_NUMPY
        ),
        media_type="application/json"
    )