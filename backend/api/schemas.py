from pydantic import BaseModel, Field
from typing import List

class StateVector(BaseModel):
    # [id, is_debris, rx, ry, rz, vx, vy, vz]
    id: int = Field(..., description="Unique object identifier")
    is_debris: bool = Field(..., description="True if object is non-maneuverable")
    rx: float = Field(..., description="ECI X-position (km)")
    ry: float = Field(..., description="ECI Y-position (km)")
    rz: float = Field(..., description="ECI Z-position (km)")
    vx: float = Field(..., description="ECI X-velocity (km/s)")
    vy: float = Field(..., description="ECI Y-velocity (km/s)")
    vz: float = Field(..., description="ECI Z-velocity (km/s)")

class Maneuver(BaseModel):
    satellite_id: int
    burn_vector_eci: List[float] = Field(..., min_items=3, max_items=3)

class TickRequest(BaseModel):
    current_time: float = Field(..., ge=0, description="Simulation epoch in seconds")
    states: List[List[float]] = Field(..., description="Batch of state vectors")

class TickResponse(BaseModel):
    time: float
    maneuvers: List[Maneuver]