import numpy as np
import orjson
from fastapi import APIRouter, Request, Response

router = APIRouter(prefix="/api")

# --- STATIC MEMORY POOL ---
# Pre-allocated once at startup. Never re-assigned.
TELEMETRY_BUFFER = np.zeros((10000, 8), dtype=np.float64)

@router.post("/tick")
async def process_tick(request: Request):
    global TELEMETRY_BUFFER
    
    # Fast-load raw JSON
    raw_body = await request.body()
    payload = orjson.loads(raw_body)
    
    # Convert incoming list to temporary numpy array
    incoming_data = np.array(payload['states'], dtype=np.float64)
    
    # PRODUCTION FIX: Slice assignment to preserve the pre-allocated buffer
    # This prevents the variable from re-pointing and triggering GC
    TELEMETRY_BUFFER[:incoming_data.shape[0], :] = incoming_data
    
    # ... (Rest of the threaded RK4 and Evasion logic) ...
    
    return Response(
        content=orjson.dumps({"status": "processed"}),
        media_type="application/json"
    )

@router.get("/state")
async def get_current_state():
    # Stream the static buffer directly. 
    # High-contrast aesthetics: dark-first delivery[cite: 9, 10].
    return Response(
        content=orjson.dumps(
            {"states": TELEMETRY_BUFFER}, 
            option=orjson.OPT_SERIALIZE_NUMPY
        ),
        media_type="application/json"
    )